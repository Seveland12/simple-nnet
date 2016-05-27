(ns nnet.math-utilities)

(def my-eps
  ; This is the epsilon to use for "real number"
  ; equality comparisons 
  0.000001)

(defn my-sq
  ; simple square function. there has to be
  ; a build-in pow equivalent, right?
  [x]
  (* x x))

(defn approx-equals?
  ; simple epsilon-type real number equality comparison
  [x y]
  (<= (Math/abs (- x y)) my-eps))
