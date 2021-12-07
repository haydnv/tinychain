import tinychain as tc
import unittest

from testutils import start_host


class TestGraph(tc.graph.Graph):
    __uri__ = tc.URI(f"/test/graph")

    @classmethod
    def create(cls):
        users = tc.table.Schema(
            [tc.Column("user_id", tc.U64)],
            [tc.Column("email", tc.String, 320), tc.Column("display_name", tc.String, 100)])

        products = tc.table.Schema(
            [tc.Column("sku", tc.U64)],
            [tc.Column("name", tc.String, 256), tc.Column("price", tc.U32)])

        orders = tc.table.Schema(
            [tc.Column("order_id", tc.U64)],
            [tc.Column("user_id", tc.U64), tc.Column("sku", tc.U64), tc.Column("quantity", tc.U32)]
        ).create_index("user", ["user_id"]).create_index("product", ["sku"])

        schema = (tc.graph.Schema()
                  .create_table("users", users)
                  .create_table("products", products)
                  .create_table("orders", orders)
                  .create_edge("friends", tc.graph.edge.Schema("users.user_id", "users.user_id"))
                  .create_edge("order_products", tc.graph.edge.Schema("products.sku", "orders.sku"))
                  .create_edge("user_orders", tc.graph.edge.Schema("users.user_id", "orders.user_id")))

        return cls(schema)


class TestService(tc.Cluster):
    __uri__ = tc.URI(f"/test")

    def _configure(self):
        self.graph = tc.chain.Sync(TestGraph.create())

    @tc.put_method
    def create_user(self, user_id: tc.U64, data: tc.Map):
        return self.graph["users"].insert([user_id], [data["email"], data["display_name"]])

    @tc.put_method
    def add_friend(self, user_id: tc.U64, friend: tc.U64):
        return self.graph.add_edge("friends", user_id, friend), self.graph.add_edge("friends", friend, user_id)

    @tc.put_method
    def add_product(self, sku: tc.U64, data: tc.Map):
        return self.graph["products"].insert([sku], [data["name"], data["price"]])

    @tc.post_method
    def place_order(self, cxt, user_id: tc.U64, sku: tc.U64, quantity: tc.U64):
        cxt.orders = self.graph["orders"]
        order_id = cxt.orders.max_id() + 1
        insert = cxt.orders.insert([order_id], [user_id, sku, quantity])
        return tc.After(insert, order_id)

    @tc.get_method
    def recommend(self, txn, user_id: tc.U64):
        txn.vector = tc.tensor.Sparse.zeros([tc.graph.edge.DIM], tc.Bool)
        txn.user_ids = tc.After(txn.vector[user_id].write(True), txn.vector)
        txn.friend_ids = tc.If(
            user_id.is_some(),
            self.graph["friends"].match(txn.user_ids, 2),
            tc.error.BadRequest("invalid user ID: {{user_id}}", user_id=user_id))

        txn.order_ids = self.graph["user_orders"].forward(txn.friend_ids)
        txn.product_ids = self.graph["order_products"].forward(txn.order_ids)
        return self.graph["products"].read_vector(txn.product_ids)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_graph", [TestService], overwrite=True, cache_size="1G")

    def testTraversal(self):
        user1 = {"email": "user12345@example.com", "display_name": "user 12345"}
        self.host.put("/test/create_user", 12345, user1)

        user2 = {"email": "user23456@example.com", "display_name": "user 23456"}
        self.host.put("/test/create_user", 23456, user2)

        self.host.put("/test/add_friend", 12345, 23456)

        product1 = {"name": "widget 1", "price": 399}
        self.host.put("/test/add_product", 1, product1)

        product2 = {"name": "widget 2", "price": 499}
        self.host.put("/test/add_product", 2, product2)

        order = {"user_id": 23456, "sku": 1, "quantity": 5}
        _order_id = self.host.post("/test/place_order", order)

        recommended = self.host.get("/test/recommend", 12345)
        self.assertEqual(recommended, [[1, "widget 1", 399]])

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
