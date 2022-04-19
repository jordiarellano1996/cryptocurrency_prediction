""" With this method our class isn't dependency inverse"""
from abc import ABC


class BadShoppingBasket:
    def buy(self, shopping):
        db = BadSqlDatabase()
        db.save(shopping)
        creditCard = BadCreditCard()
        creditCard.pay(shopping)


class BadSqlDatabase:
    def save(self, shopping):
        pass


class BadCreditCard:
    def pay(self, shopping):
        pass


""" With this method our class isn't dependency inverse"""
import abc


class ShoppingBasket:
    def __init__(self, PersistenceIn, PaymentMethodIn):
        self.Persistence = PersistenceIn
        self.PaymentMethod = PaymentMethodIn

    def buy(self, shopping):
        db = self.Persistence()
        db.save(shopping)
        pay_method = self.PaymentMethod()
        pay_method.pay(shopping)


class Persistence(abc.ABC):
    @abc.abstractmethod
    def save(self, shopping):
        """ Allows to use different save methods"""


class SqlDatabase(Persistence):
    def save(self, shopping):
        print(f"Save {shopping} with SqlDatabase")


class Server(Persistence):
    def save(self, shopping):
        print(f"Save {shopping} with Server")


class PaymentMethod(abc.ABC):
    @abc.abstractmethod
    def pay(self, shopping):
        """ Allows to use different save methods"""


class CreditCard(PaymentMethod):
    def pay(self, shopping):
        print(f"Pay {shopping} with CreditCard")


class PayPal(PaymentMethod):
    def pay(self, shopping):
        print(f"Pay {shopping} with PayPal")


if __name__ == "__main__":
    sb = ShoppingBasket(SqlDatabase, CreditCard)
    sb.buy("Zapatos Nike")
    print("\n----Ejemplo2---\n")
    sb2 = ShoppingBasket(SqlDatabase, PayPal)
    sb2.buy("Zapatos Nike")
    print("\n----Ejemplo3---\n")
    sb3 = ShoppingBasket(Server, PayPal)
    sb3.buy("Zapatos Addidas")
