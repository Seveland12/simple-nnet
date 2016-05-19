(ns nnet.nnet
  (:require (incanter [core :refer :all]))) 

;(use '(incanter core stats charts io))


(def A (matrix [[1 2 3] [4 5 6] [7 8 9]]))


(def v (vector 1 2 3))


(def wh (matrix [[0.362985 0.418378] [-0.464489 -0.554121] [-0.720958 0.504430]]))
(def wo (vector [0.620124 -0.446396 0.692502]))

; input-vector has an extra final component = 1.0 to accomodate the
; bias terms
(def input-vector
  ; input-vector has an extra final component = 1.0 to accomodate the
  ; bias terms
  (trans (matrix [0.5 0.5 1.0])))

(defn activation-function
  ;This is the sigmoid activation function used by each individual neuron.
  ;This version scales the tanh function to saturate at yyyy and have its 
  ;maximal derivative at +- xxxx as suggested in Haykin.
  [x]
  (* 1.7159 (Math/tanh (* 0.6666 x))))
  
(defn calculate-hidden-layer
  [i w]
  (map activation-function (mmult i w)))

(def q (activation-function 1.2))


(def abc (plus wh wh))















