import tinychain as tc
import unittest

from testutils import start_host


class TestApp(tc.Graph):
    __uri__ = tc.URI("/test/graph")

    def _schema(self):
        users = tc.schema.Table(
            [tc.Column("user_id", tc.U64)],
            [tc.Column("email", tc.String, 320), tc.Column("display_name", tc.String, 100)])

        products = tc.schema.Table(
            [tc.Column("sku", tc.U64)],
            [tc.Column("name", tc.String, 256), tc.Column("price", tc.U32)])

        orders = tc.schema.Table(
            [tc.Column("order_id", tc.U64)],
            [tc.Column("user_id", tc.Bytes, 16), tc.Column("sku", tc.U32), tc.Column("quantity", tc.UInt)])

        schema = (tc.schema.Graph(tc.chain.Block)
                  .create_table("users", users)
                  .create_table("products", products)
                  .create_table("orders", orders)
                  .create_edge("friends", tc.schema.Edge("users.user_id", "users.user_id"))
                  .create_edge("sku", tc.schema.Edge("products.sku", "orders.sku"))
                  .create_edge("orders", tc.schema.Edge("users.user_id", "orders.user_id")))

        return schema

    @tc.post_method
    def add_product(self, sku: tc.U64, name: tc.String, price: tc.U32):
        return self.products.insert([sku], [name, price])

    @tc.post_method
    def create_user(self, user_id: tc.U64, email: tc.String, display_name: tc.String):
        return self.users.insert([user_id], [email, display_name])

    @tc.put_method
    def add_friend(self, user_id: tc.U64, friend: tc.U64):
        return self.add_edge("friends", (user_id, friend)), self.add_edge("friends", (friend, user_id))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [TestApp], overwrite=True)

    def testGraphTraversal(self):
        user = {"user_id": 12345, "email": "user12345@example.com", "display_name": "user 12345"}
        self.host.post("/test/graph/create_user", user)

        user = {"user_id": 23456, "email": "user23456@example.com", "display_name": "user 23456"}
        self.host.post("/test/graph/create_user", user)

        self.host.put("/test/graph/add_friend", 12345, 23456)

        product = {"sku": 1, "name": "widget 1", "price": 399}
        self.host.post("/test/graph/add_product", product)

        product = {"sku": 2, "name": "widget 2", "price": 499}
        self.host.post("/test/graph/add_product", product)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
