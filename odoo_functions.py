

import xmlrpc.client


class odoo():
    def __init__(self, url, db , username,password):
        self.url = url
        self.db = db
        self.username = username
        self.password = password
        self.common = xmlrpc.client.ServerProxy('{}/xmlrpc/2/common'.format(url))
        self.uid = self.common.authenticate(db, username, password, {})
        self.models = xmlrpc.client.ServerProxy('{}/xmlrpc/2/object'.format(url))


    def search_products(self,product_name: str):
        return self.models.execute_kw(self.db, self.uid, self.password, 'product.product', 'search_read', [[['name', 'ilike', product_name]]], {'fields': ['name', 'list_price'], 'limit': 5})

    def place_order(self,product_name: str , quantity: int):
        return f"{quantity} {product_name} order is placed"
         
