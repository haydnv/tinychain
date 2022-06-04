import tinychain as tc

URI = tc.URI("/test/tctest/client")


class Product(tc.app.Model):
    __uri__ = URI.append("Product")

    price = tc.Column("price", tc.I32)
    name = tc.Column("name", tc.String, 100)

    def __init__(self, product, name, price):
        self.product = product
        self.price = price
        self.name = name


class User(tc.app.Model):
    __uri__ = URI.append("User")

    first_name = tc.Column("first_name", tc.String, 100)
    last_name = tc.Column("last_name", tc.String, 100)

    def __init__(self, user_id, first_name, last_name):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name


class Order(tc.app.Model):
    __uri__ = URI.append("Product")

    quantity = tc.Column("quantity", tc.I32)
    product_id = Product
    user_id = User

    def __init__(self, order_id, quantity, user_id, product_id):
        self.quantity = quantity
        self.user_id = user_id
        self.product_id = product_id
