(ns nnet.math-utilities
  (:require [clojure.math.numeric-tower :as cljmath]))

(def my-eps
  ; This is the epsilon to use for "real number"
  ; equality comparisons 
  0.0001)

(defn my-sq
  ; simple square function. there has to be
  ; a build-in pow equivalent, right?
  [x]
  (cljmath/expt x 2))

(defn approx-equals?
  ; simple epsilon-type real number equality comparison
  [x y]
  (<= (cljmath/abs (- x y)) my-eps))

(defn n-ones-and-a-zero
  ; Returns a vector of n ones followed by one zero.
  ; Used to construct an nxn identity matrix with
  ; its last 1 zeroed out.
  [n]
  (conj (vec (take n (repeat 1))) 0.0))
