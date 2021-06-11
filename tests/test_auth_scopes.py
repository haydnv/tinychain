import unittest
import tinychain as tc

from testutils import start_host


IN_STOCK = tc.Int(100)
SCOPE = "/buy"


class Wholesaler(tc.Cluster):
    __uri__ = tc.URI("/app/wholesaler")

    def _configure(self):
        self.in_stock = tc.chain.Sync(IN_STOCK)

    @tc.put_method
    def update(self, _txn, _key: tc.Nil, new_inventory: tc.UInt):
        # TODO: add an auth scope, remove key param
        return self.in_stock.set(new_inventory)

    @tc.post_method
    def buy(self, txn, quantity: tc.UInt):
        txn.inventory = self.inventory()
        txn.new_inventory = txn.inventory - quantity
        txn.sale = tc.If(
            quantity > txn.inventory,
            tc.error.BadRequest("requested quantity is unavailable"),
            self.update(None, txn.new_inventory))

        return tc.After(self.authorize(SCOPE), txn.sale)

    @tc.get_method
    def inventory(self) -> tc.Number:
        return self.in_stock


class Retailer(tc.Cluster):
    __uri__ = tc.URI("/app/retailer")

    @tc.post_method
    def buy(self, _txn, quantity: tc.Number):
        wholesaler = tc.use(Wholesaler)
        op = tc.post_op(lambda txn, quantity: wholesaler.buy(quantity=quantity))
        return self.grant(SCOPE, op, quantity=quantity)


class InteractionTests(unittest.TestCase):
    def testWorkflow(self):
        host = start_host("test_auth_scopes", [Wholesaler, Retailer])

        actual = host.get("/app/wholesaler/inventory")
        self.assertEqual(100, actual)

        with self.assertRaises(tc.error.Unauthorized):
            host.post("/app/retailer/buy", {"quantity": 10})

        host.put("/app/wholesaler/install", "http://127.0.0.1:8702" + tc.uri(Retailer), ["buy"])

        host.post("/app/retailer/buy", {"quantity": 10})
        self.assertEqual(90, host.get("/app/wholesaler/inventory"))


if __name__ == "__main__":
    unittest.main()

