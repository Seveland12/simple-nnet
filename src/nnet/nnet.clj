(ns nnet.nnet
  (:require (incanter [core :refer :all]))) 

(def wh (matrix [[0.362985 0.418378] [-0.464489 -0.554121] [-0.720958 0.504430]]))
(def wo (matrix [0.620124 -0.446396 0.692502]))

; input-vector has an extra final component = 1.0 to accomodate the
; bias terms
(def input-vector
  ; input-vector has an extra final component = 1.0 to accomodate the
  ; bias terms
  (trans (matrix [0.5 -0.5 1.0])))

(def desired-response
  (matrix [0.5]))

(defn activation-function
  ;This is the sigmoid activation function used by each individual neuron.
  ;This version scales the tanh function to saturate at yyyy and have its 
  ;maximal derivative at +- xxxx as suggested in Haykin.
  [x]
  (* 1.7159 (Math/tanh (* 0.6666 x))))
  
(defn calculate-hidden-layer
  [i w]
  (matrix (conj (mapv activation-function (mmult i w)) 1.0)))


(defn calculate-output-layer
  [h w]
  (matrix (mapv activation-function (mmult (trans h) w))))


(defn evaluate-network
  [i]
  (calculate-output-layer (calculate-hidden-layer i wh) wo))

(defn train
  []
  (let [current-output (evaluate-network input-vector)]
    (let [current-error (minus desired-response current-output)]
            (print current-error))))










