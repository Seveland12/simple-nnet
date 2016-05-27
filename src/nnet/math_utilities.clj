(ns nnet.math-utilities)

(def my-eps 0.000001)

(defn my-sq [x] (* x x))

(defn approx-equals?
  [x y]
  (<= (Math/abs (- x y)) my-eps))
