
# Parametric Activation Function Applied to XOR Problem

**Description:**  
This experiment explores the use of a parametric hyperbolic tangent activation function (with learnable alpha and beta weights) for solving the XOR problem using a neural network. The performance of the model using this parametric activation function is compared to the standard version without it. The results show that the model with the parametric activation function learns faster.

**abTanh Class Code (for XOR problem):**
```python
class abTanh(layers.Layer):
    def __init__(self, units=None, init_a=7, init_b=0, **kwargs):
        super(abTanh, self).__init__(**kwargs)
        self.units = units
        self.init_a = init_a
        self.init_b = init_b

    def build(self, input_shape):
        units_shape = self.units if self.units else input_shape[1:]
        self.b = self.add_weight(shape=units_shape, initializer=tf.constant_initializer(self.init_b), trainable=True, name="b")
        self.a = self.add_weight(shape=units_shape, initializer=tf.constant_initializer(self.init_a), trainable=True, name="a")

    def call(self, inputs, **kwargs):
        x = tf.subtract(inputs, self.b)
        x = tf.nn.tanh(x)
        x = tf.multiply(x, self.a)
        return x
```

**Images:**
![Experiment 1](image1.gif)  
![Experiment 2](image2.gif)

---

# Parametric ReLU Function in Transformer

**Description:**  
This project investigates the use of the Parametric ReLU (PReLU) activation function in a Transformer model. The goal is to evaluate how the learnable parameters (alpha and beta) affect the learning efficiency and performance of Transformer-based architectures.

**abRelu Class Code:**
```python
class abRelu(tf.keras.layers.Layer):
    def __init__(self, initial_a=6.0, initial_b=0.0):
        super(abRelu, self).__init__()
        self.initial_a = initial_a
        self.initial_b = initial_b

    def build(self, input_shape):
        # Create trainable parameters a and b
        self.a = self.add_weight(
            shape=(input_shape[-1],),  # Shape for each feature in the input
            initializer=tf.keras.initializers.Constant(self.initial_a),
            trainable=True,
            name='a'
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),  # Shape for each feature in the input
            initializer=tf.keras.initializers.Constant(self.initial_b),
            trainable=True,
            name='b'
        )
        super(abRelu, self).build(input_shape)

    def call(self, inputs):
        return tf.where(inputs >= 0, self.a * (inputs - self.b), 0)

    def get_config(self):
        config = super(abRelu, self).get_config()
        config.update({
            'initial_a': self.initial_a,
            'initial_b': self.initial_b
        })
        return config
```


**Results:**
*Insert result graph here*
```

