from layer_naive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 順伝播
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(f"合計金額は {int(price)} 円")

# 逆伝播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f"リンゴの値段に対する微分は {dapple}")
print(f"リンゴの個数に対する微分は {int(dapple_num)}")
print(f"消費税に対する微分は {dtax}")