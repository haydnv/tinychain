import unittest
import tinychain as tc

from testutils import start_host


IN_STOCK = 100


class Producer(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/producer")

    def configure(self):
        self.in_stock = tc.Chain.Sync(IN_STOCK)

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        txn.new_inventory = self.inventory() - quantity

        return tc.If(
            txn.new_inventory < 0,
            tc.error.BadRequest("requested quantity is unavailable"),
            self.in_stock.set(txn.new_inventory))

    @tc.get_method
    def inventory(self) -> tc.Number:
        return self.in_stock.subject()


class Wholesaler(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/wholesaler")

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        producer = tc.use(Producer)
        return producer.buy(quantity=quantity)


class Retailer(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/retailer")

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        wholesaler = tc.use(Wholesaler)
        return wholesaler.buy(quantity=quantity)


class InteractionTests(unittest.TestCase):
    def testWorkflow(self):
        host = start_host("test_yc_demo", [Producer, Wholesaler, Retailer])

        actual = host.get("/app/producer/inventory")
        self.assertEqual(IN_STOCK, actual)

#        host.post("/app/wholesaler/buy", 10)
#        self.assertEqual(90, host.get("/app/producer/inventory"))


if __name__ == "__main__":
    unittest.main()

