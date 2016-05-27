(ns nnet.nnet
  (:require (incanter [core :refer :all]))) 

;; (defrecord NeuralNet [hidden-weights output-weights])

;; (def wh (matrix [[0.362985 0.418378 0.0]
;;                  [-0.464489 -0.554121 0.0]
;;                  [-0.720958 0.504430 1.0]]))
;; (def wo (matrix [0.620124 -0.446396 0.692502]))

;; (defn activation-function
;;   ;This is the sigmoid activation function used by each individual neuron.
;;   ;This version scales the tanh function to saturate at yyyy and have its 
;;   ;maximal derivative at +- xxxx as suggested in Haykin.
;;   [x]
;;   (* 1.7159 (Math/tanh (* 0.6666 x))))

;; (defn hidden-layer
;;   [i w]
;;   (matrix (mapv activation-function (mmult i w))))

;; (defn output-layer
;;   [h w]
;;   (matrix (mapv activation-function (mmult (trans h) w))))

;; (defn evaluate-network
;;   [i]
;;   (output-layer (hidden-layer i wh) wo))
