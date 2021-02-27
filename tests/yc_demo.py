import unittest
import tinychain as tc

from testutils import start_host


IN_STOCK = 100
SCOPE = "/buy"


class Producer(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/producer")

    def configure(self):
        self.in_stock = tc.Chain.Sync(IN_STOCK)

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        txn.inventory = self.inventory()
        txn.new_inventory = txn.inventory - quantity
        txn.sale = tc.If(
            quantity > self.inventory(),
            tc.error.BadRequest("requested quantity is unavailable"),
            self.in_stock.set(txn.new_inventory))

        return tc.After(self.authorize(SCOPE), txn.sale)

    @tc.get_method
    def inventory(self) -> tc.Number:
        return self.in_stock.subject()


class Wholesaler(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/wholesaler")

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        producer = tc.use(Producer)

        return self.grant(
            SCOPE,
            {"/state/scalar/op/post": [("buy_result", producer.buy(quantity=quantity))]},
            {"quantity": quantity},
        )


class InteractionTests(unittest.TestCase):
    def testWorkflow(self):
        host = start_host("test_yc_demo", [Producer, Wholesaler])

        actual = host.get("/app/producer/inventory")
        self.assertEqual(IN_STOCK, actual)

        with self.assertRaises(tc.error.Unauthorized):
            host.post("/app/wholesaler/buy", {"quantity": 10})

        host.put("/app/producer/install", "http://127.0.0.1:8702" + tc.uri(Wholesaler), ["buy"])

        host.post("/app/wholesaler/buy", {"quantity": 10})
        self.assertEqual(90, host.get("/app/producer/inventory"))


if __name__ == "__main__":
    unittest.main()

