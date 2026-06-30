"""
UPI Transaction Network Generator
==================================
Generates a synthetic but realistic graph of UPI (Unified Payments Interface)
transactions between accounts, embedding three classic fraud-ring topologies
that show up in real Indian fintech fraud:

1. MULE FAN-IN/FAN-OUT  : many "source" accounts send small amounts into a
   single mule account, which then rapidly fans the money back out to a
   handful of "cash-out" accounts (classic money-mule layering pattern).
2. CIRCULAR LOOPS        : money moves A -> B -> C -> D -> A in a short
   cycle, often used to fake transaction velocity / wash money / game
   merchant cashback or credit-scoring systems.
3. SMURFING STARS        : one controller account splits a large sum into
   many small sub-threshold transfers to avoid AML reporting limits, fanned
   across a star of dummy accounts, which later consolidate back.

Normal users transact in a sparse, mostly-tree-like / random small-world
pattern with realistic amount distributions (UPI transactions are typically
small: groceries, rent, P2P transfers, bill payments).

Output: a directed, weighted, time-stamped NetworkX MultiDiGraph plus a
flat pandas DataFrame of transactions (the kind a real UPI switch / NPCI
log would produce), with ground-truth fraud labels on both edges and nodes.
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

RNG_SEED = 42


class UPITransactionGenerator:
    def __init__(self, seed=RNG_SEED):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.G = nx.MultiDiGraph()
        self.transactions = []
        self.node_counter = 0
        self.start_time = datetime(2025, 1, 1)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _new_account(self, label_prefix="U"):
        acc_id = f"{label_prefix}{self.node_counter:06d}"
        self.node_counter += 1
        return acc_id

    def _rand_time(self, day_window=180):
        return self.start_time + timedelta(
            days=random.randint(0, day_window),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

    def _add_txn(self, src, dst, amount, ts, is_fraud, ring_id=None, pattern=None):
        self.G.add_edge(src, dst, amount=amount, timestamp=ts,
                         is_fraud=is_fraud, ring_id=ring_id, pattern=pattern)
        self.transactions.append(dict(
            src=src, dst=dst, amount=round(amount, 2), timestamp=ts,
            is_fraud=is_fraud, ring_id=ring_id, pattern=pattern
        ))

    # ------------------------------------------------------------------ #
    # 1. Legit traffic
    # ------------------------------------------------------------------ #
    def generate_legit_traffic(self, n_accounts=2000, n_txns=9000):
        accounts = [self._new_account("U") for _ in range(n_accounts)]
        for a in accounts:
            self.G.add_node(a, is_fraud_node=0, account_type="genuine")

        # realistic UPI amount distribution: lots of small P2P/merchant txns,
        # occasional larger rent/bill payments
        for _ in range(n_txns):
            src, dst = random.sample(accounts, 2)
            amount = float(np.random.choice(
                [np.random.uniform(10, 500),
                 np.random.uniform(500, 5000),
                 np.random.uniform(5000, 50000)],
                p=[0.7, 0.25, 0.05]
            ))
            ts = self._rand_time()
            self._add_txn(src, dst, amount, ts, is_fraud=0, pattern="legit")
        return accounts

    # ------------------------------------------------------------------ #
    # 2. Mule fan-in / fan-out rings
    # ------------------------------------------------------------------ #
    def generate_mule_rings(self, n_rings=15, feeders_per_ring=(8, 20),
                             cashouts_per_ring=(2, 5)):
        for r in range(n_rings):
            ring_id = f"MULE_{r}"
            mule = self._new_account("M")
            self.G.add_node(mule, is_fraud_node=1, account_type="mule")

            n_feed = random.randint(*feeders_per_ring)
            n_cash = random.randint(*cashouts_per_ring)
            feeders = [self._new_account("F") for _ in range(n_feed)]
            cashouts = [self._new_account("C") for _ in range(n_cash)]
            for f in feeders:
                self.G.add_node(f, is_fraud_node=1, account_type="feeder")
            for c in cashouts:
                self.G.add_node(c, is_fraud_node=1, account_type="cashout")

            base_time = self._rand_time()
            # fan-in: many feeders -> mule, in a tight time burst (structuring)
            total_in = 0
            for f in feeders:
                amt = np.random.uniform(1500, 9500)  # just under common AML eyeball thresholds
                total_in += amt
                ts = base_time + timedelta(minutes=random.randint(0, 90))
                self._add_txn(f, mule, amt, ts, is_fraud=1, ring_id=ring_id, pattern="mule_fanin")

            # rapid fan-out: mule -> cashout accounts within minutes/hours (layering)
            remaining = total_in * random.uniform(0.85, 0.98)  # small cut/fees
            for i, c in enumerate(cashouts):
                share = remaining / n_cash * random.uniform(0.7, 1.3)
                ts = base_time + timedelta(minutes=random.randint(60, 240))
                self._add_txn(mule, c, share, ts, is_fraud=1, ring_id=ring_id, pattern="mule_fanout")

    # ------------------------------------------------------------------ #
    # 3. Circular loop rings (wash trading / fake velocity)
    # ------------------------------------------------------------------ #
    def generate_circular_loops(self, n_rings=12, loop_len_range=(3, 6)):
        for r in range(n_rings):
            ring_id = f"LOOP_{r}"
            loop_len = random.randint(*loop_len_range)
            nodes = [self._new_account("L") for _ in range(loop_len)]
            for n in nodes:
                self.G.add_node(n, is_fraud_node=1, account_type="loop_node")

            base_amt = np.random.uniform(8000, 60000)
            base_time = self._rand_time()
            # money cycles through the ring 1-3 times, amount shrinking slightly each hop (fees)
            n_cycles = random.randint(1, 3)
            t = base_time
            amt = base_amt
            for cyc in range(n_cycles):
                for i in range(loop_len):
                    src = nodes[i]
                    dst = nodes[(i + 1) % loop_len]
                    t = t + timedelta(minutes=random.randint(5, 45))
                    amt *= random.uniform(0.95, 0.99)
                    self._add_txn(src, dst, amt, t, is_fraud=1, ring_id=ring_id, pattern="circular_loop")

    # ------------------------------------------------------------------ #
    # 4. Smurfing star (structuring to avoid reporting thresholds)
    # ------------------------------------------------------------------ #
    def generate_smurfing_stars(self, n_rings=10, n_smurfs_range=(10, 25)):
        for r in range(n_rings):
            ring_id = f"SMURF_{r}"
            controller = self._new_account("S")
            collector = self._new_account("S")
            self.G.add_node(controller, is_fraud_node=1, account_type="smurf_controller")
            self.G.add_node(collector, is_fraud_node=1, account_type="smurf_collector")

            n_smurfs = random.randint(*n_smurfs_range)
            smurfs = [self._new_account("S") for _ in range(n_smurfs)]
            for s in smurfs:
                self.G.add_node(s, is_fraud_node=1, account_type="smurf")

            base_time = self._rand_time()
            for s in smurfs:
                amt = np.random.uniform(1000, 9999)  # deliberately below 10k-style thresholds
                t1 = base_time + timedelta(minutes=random.randint(0, 30))
                self._add_txn(controller, s, amt, t1, is_fraud=1, ring_id=ring_id, pattern="smurf_out")
                t2 = t1 + timedelta(minutes=random.randint(10, 120))
                self._add_txn(s, collector, amt * random.uniform(0.97, 1.0), t2,
                              is_fraud=1, ring_id=ring_id, pattern="smurf_in")

    # ------------------------------------------------------------------ #
    def add_realism_noise(self, all_accounts, fraud_accounts):
        """
        Make the classification task realistic instead of trivially separable:
        1) Fraud-ring accounts also transact normally sometimes (mules have real lives too).
        2) A handful of genuine accounts behave like high-degree 'hubs' (popular merchants,
           UPI collection accounts) which superficially resemble fan-in patterns but are NOT fraud.
        3) A few genuine accounts participate in incidental short cycles by chance (A pays B,
           B refunds A days later, etc.) without being part of an actual ring.
        """
        # 1) ~30% of fraud accounts also send/receive a couple of normal txns
        sample_fraud = random.sample(fraud_accounts, k=int(len(fraud_accounts) * 0.3))
        for acc in sample_fraud:
            for _ in range(random.randint(1, 3)):
                other = random.choice(all_accounts)
                if other == acc:
                    continue
                amt = np.random.uniform(20, 800)
                ts = self._rand_time()
                if random.random() < 0.5:
                    self._add_txn(acc, other, amt, ts, is_fraud=0, pattern="legit_by_fraud_acct")
                else:
                    self._add_txn(other, acc, amt, ts, is_fraud=0, pattern="legit_by_fraud_acct")

        # 2) genuine high-degree hub accounts (e.g. popular merchant / electricity board)
        hubs = random.sample(all_accounts, k=12)
        for hub in hubs:
            n_payers = random.randint(15, 40)
            base_time = self._rand_time()
            for _ in range(n_payers):
                payer = random.choice(all_accounts)
                if payer == hub:
                    continue
                amt = np.random.uniform(50, 2000)
                ts = base_time + timedelta(hours=random.randint(0, 24 * 10))
                self._add_txn(payer, hub, amt, ts, is_fraud=0, pattern="legit_merchant_hub")

        # 3) incidental genuine 2-3 hop "cycles" (pay + later unrelated refund) - small amount, spaced far apart in time
        for _ in range(25):
            a, b = random.sample(all_accounts, 2)
            amt1 = np.random.uniform(100, 1500)
            t1 = self._rand_time()
            self._add_txn(a, b, amt1, t1, is_fraud=0, pattern="legit_incidental")
            t2 = t1 + timedelta(days=random.randint(5, 60))  # far apart, not a velocity pattern
            amt2 = np.random.uniform(100, 1500)
            self._add_txn(b, a, amt2, t2, is_fraud=0, pattern="legit_incidental_return")

    def build(self):
        print("Generating legitimate UPI traffic...")
        legit_accounts = self.generate_legit_traffic()
        print("Embedding mule fan-in/fan-out rings...")
        self.generate_mule_rings()
        print("Embedding circular transaction loops...")
        self.generate_circular_loops()
        print("Embedding smurfing star patterns...")
        self.generate_smurfing_stars()

        fraud_accounts = [n for n, d in self.G.nodes(data=True) if d.get("is_fraud_node") == 1]
        print("Adding behavioural realism / noise...")
        self.add_realism_noise(legit_accounts, fraud_accounts)

        df = pd.DataFrame(self.transactions).sort_values("timestamp").reset_index(drop=True)
        df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]
        print(f"Done. {self.G.number_of_nodes()} accounts, {self.G.number_of_edges()} transactions, "
              f"{df['is_fraud'].mean()*100:.2f}% fraud-labelled edges.")
        return self.G, df


if __name__ == "__main__":
    gen = UPITransactionGenerator()
    G, df = gen.build()
    df.to_csv("/home/claude/upi_fraud_graph/data/transactions.csv", index=False)
    print(df.head())
