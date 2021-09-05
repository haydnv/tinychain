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
            [tc.Column("user_id", tc.U64), tc.Column("sku", tc.U64), tc.Column("quantity", tc.U32)]
        ).create_index("user", ["user_id"]).create_index("product", ["sku"])

        schema = (tc.schema.Graph(tc.chain.Block)
                  .create_table("users", users)
                  .create_table("products", products)
                  .create_table("orders", orders)
                  .create_edge("friends", tc.schema.Edge("users.user_id", "users.user_id"))
                  .create_edge("order_products", tc.schema.Edge("products.sku", "orders.sku"))
                  .create_edge("user_orders", tc.schema.Edge("users.user_id", "orders.user_id")))

        return schema

    @tc.put_method
    def add_product(self, sku: tc.U64, data: tc.Map):
        return self.products.insert([sku], [data["name"], data["price"]])

    @tc.put_method
    def create_user(self, user_id: tc.U64, data: tc.Map):
        return self.users.insert([user_id], [data["email"], data["display_name"]])

    @tc.put_method
    def add_friend(self, user_id: tc.U64, friend: tc.U64):
        return self.add_edge("friends", user_id, friend), self.add_edge("friends", friend, user_id)

    @tc.post_method
    def place_order(self, user_id: tc.U64, sku: tc.U64, quantity: tc.U64):
        order_id = self.orders.max_id() + 1
        return tc.After(self.orders.insert([order_id], [user_id, sku, quantity]), order_id)

    @tc.get_method
    def recommend(self, txn, user_id: tc.U64):
        txn.vector = tc.tensor.Sparse.zeros([tc.I64.max()], tc.Bool)
        txn.user_ids = tc.After(txn.vector.write([user_id], True), txn.vector)
        txn.friend_ids = tc.If(
            user_id.is_some(),
            self.friends.match(txn.user_ids, 2),
            tc.error.BadRequest("invalid user ID: {{user_id}}", user_id=user_id))

        txn.order_ids = self.user_orders.forward(txn.friend_ids)
        txn.product_ids = self.order_products.forward(txn.order_ids)
        return self.products.read_vector(txn.product_ids)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [TestApp], overwrite=True, cache_size="1G")

    def testGraphTraversal(self):
        user1 = {"email": "user12345@example.com", "display_name": "user 12345"}
        self.host.put("/test/graph/create_user", 12345, user1)

        user2 = {"email": "user23456@example.com", "display_name": "user 23456"}
        self.host.put("/test/graph/create_user", 23456, user2)

        self.host.put("/test/graph/add_friend", 12345, 23456)

        product1 = {"name": "widget 1", "price": 399}
        self.host.put("/test/graph/add_product", 1, product1)

        product2 = {"name": "widget 2", "price": 499}
        self.host.put("/test/graph/add_product", 2, product2)

        order = {"user_id": 23456, "sku": 1, "quantity": 5}
        _order_id = self.host.post("/test/graph/place_order", order)

        recommended = self.host.get("/test/graph/recommend", 12345)
        self.assertEqual(recommended, [[1, "widget 1", 399]])

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
