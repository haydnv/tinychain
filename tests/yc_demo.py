import unittest
import tinychain as tc

from testutils import start_host


QUANTITY = tc.Number(100)


class Producer(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/producer")

    def configure(self):
        self.weight = tc.Chain.Sync(QUANTITY)

    @tc.post_method
    def buy(self, txn, quantity: tc.Number):
        txn.new_inventory = self.inventory() - quantity

        return tc.If(
            txn.new_inventory < 0,
            tc.error.BadRequest("requested quantity is unavailable"),
            self.inventory() - quantity)

    @tc.get_method
    def inventory(self) -> tc.Number:
        return self.weight.subject()


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
    def testStartup(self):
        host = start_host("test_interaction", [Producer, Wholesaler, Retailer])

        actual = host.get("/app/producer")
        self.assertEqual(QUANTITY, actual)


if __name__ == "__main__":
    unittest.main()

