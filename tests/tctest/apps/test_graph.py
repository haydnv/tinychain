import unittest
import time
import tinychain as tc

from ..process import start_host

URI = tc.URI("/test/graph")


class TestGraph(tc.graph.Graph):
    __uri__ = URI

    @tc.put
    def create_user(self, user_id: tc.U32, data: tc.Map):
        return self.user.insert([user_id], [data["email"], data["display_name"]])

    @tc.put
    def add_friend(self, user_id: tc.U32, friend: tc.U32):
        return self.friend.link(user_id, friend), self.friend.link(friend, user_id)

    @tc.put
    def add_product(self, sku: tc.U32, data: tc.Map):
        return self.product.insert([sku], [data["name"], data["price"]])

    @tc.post
    def place_order(self, user_id: tc.U32, sku: tc.U32, quantity: tc.U32):
        order_id = self.order.max_id() + 1
        insert = self.order.insert([order_id], [user_id, sku, quantity])
        return tc.after(insert, order_id)

    @tc.get
    def recommend(self, txn, user_id: tc.U32):
        txn.vector = tc.graph.Vector.create()
        txn.user_ids = tc.after(txn.vector[user_id].write(True), txn.vector)

        txn.friend_ids = tc.If(
            user_id.is_some(),
            self.friend.match(txn.user_ids, 2),
            tc.error.BadRequest("invalid user ID: {{user_id}}", user_id=user_id))

        txn.order_ids = self.user_order.foreign(txn.friend_ids)
        txn.product_ids = self.order_product.foreign(txn.order_ids)
        return self.product.read_vector(txn.product_ids)


@unittest.skip  # TODO: fix and re-enable
class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        users = tc.table.Schema(
            [tc.Column("user_id", tc.U32)],
            [tc.Column("email", tc.String, 320), tc.Column("display_name", tc.String, 100)])

        products = tc.table.Schema(
            [tc.Column("sku", tc.U32)],
            [tc.Column("name", tc.String, 256), tc.Column("price", tc.U32)])

        orders = tc.table.Schema(
            [tc.Column("order_id", tc.U32)],
            [tc.Column("user_id", tc.U32), tc.Column("sku", tc.U32), tc.Column("quantity", tc.U32)]
        ).create_index("user", ["user_id"]).create_index("product", ["sku"])

        schema = (tc.graph.Schema()
                  .create_table("user", users)
                  .create_table("product", products)
                  .create_table("order", orders)
                  .create_edge("friend", tc.graph.edge.Schema("user.user_id", "user.user_id"))
                  .create_edge("order_product", tc.graph.edge.Schema("product.sku", "order.sku"))
                  .create_edge("user_order", tc.graph.edge.Schema("user.user_id", "order.user_id")))

        cls.host = start_host("test_graph", [TestGraph(schema=schema)])

    def testTraversal(self):
        user1 = {"email": "user12345@example.com", "display_name": "user 12345"}
        self.host.put(URI.append("create_user"), 12345, user1)

        user2 = {"email": "user23456@example.com", "display_name": "user 23456"}
        self.host.put(URI.append("create_user"), 23456, user2)

        self.host.put(URI.append("add_friend"), 12345, 23456)

        product1 = {"name": "widget 1", "price": 399}
        self.host.put(URI.append("add_product"), 1, product1)

        product2 = {"name": "widget 2", "price": 499}
        self.host.put(URI.append("add_product"), 2, product2)

        order = {"user_id": 23456, "sku": 1, "quantity": 5}
        _order_id = self.host.post(URI.append("place_order"), order)

        start = time.time()
        recommended = self.host.get(URI.append("recommend"), 12345)
        self.assertEqual(recommended, [[1, "widget 1", 399]])
        elapsed = (time.time() - start) * 1000
        print(f"breadth-first graph traversal ran in {elapsed:.2f} ms")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
