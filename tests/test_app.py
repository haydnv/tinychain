import tinychain as tc
import unittest

from testutils import start_host


class TestApp(tc.Graph):
    __uri__ = tc.URI("/test/graph")

    def _schema(self):
        users = tc.schema.Table(
            [tc.Column("user_id", tc.Bytes, 16)],
            [tc.Column("email", tc.String, 320), tc.Column("display_name", tc.String, 100)])

        products = tc.schema.Table(
            [tc.Column("sku", tc.U32)],
            [tc.Column("name", tc.String, 256), tc.Column("price", tc.U32)])

        orders = tc.schema.Table(
            [tc.Column("order_id", tc.Bytes, 16)],
            [tc.Column("user_id", tc.Bytes, 16), tc.Column("sku", tc.U32), tc.Column("quantity", tc.UInt)])

        schema = (tc.schema.Graph(tc.chain.Block)
                  .create_table("users", users)
                  .create_table("products", products)
                  .create_table("orders", orders)
                  .create_edge("sku", "products.sku", "orders.sku")
                  .create_edge("orders", "users.user_id", "orders.user_id"))

        return schema

    @tc.post_method
    def add_product(self, sku: tc.U32, name: tc.String, price: tc.U32):
        return self.products.insert([sku], [name, price])

    @tc.post_method
    def create_user(self, user_id: tc.Bytes, email: tc.String, display_name: tc.String):
        return self.users.insert([user_id], [email, display_name])


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [TestApp], overwrite=True)

    def testGraphTraversal(self):
        user = {"user_id": "MTIzNDU=", "email": "user12345@example.com", "display_name": "user 12345"}
        self.host.post("/test/graph/create_user", user)

        product = {"sku": 12345, "name": "widget", "price": 399}
        self.host.post("/test/graph/add_product", product)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
