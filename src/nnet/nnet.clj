(ns nnet.nnet
  (:require [incanter [core :refer :all]]
            [nnet.data-structures :refer :all])
  (:use [nnet.math-utilities :as utils :only [approx-equals?
                                              my-sq]])) 

(defn activation-function
  ;This is the sigmoid activation function used by each individual neuron.
  ;This version scales the tanh function to saturate at yyyy and have its 
  ;maximal derivative at +- xxxx as suggested in Haykin.
  [x]
  (* 1.7159 (Math/tanh (* 0.6666 x))))

(defn activation-function-deriv
  ; Clearly this is the derivative of the activation function.
  ; Hard-coded for now.
  [x]
  (/ 0.1439333 (utils/my-sq (Math/cosh (* 0.66666 x)))))

(defn number-of-input-neurons
  ; Returns the number of input neurons in NeuralNet net
  [net]
  (- (nrow (.hidden-weights net)) 1))

(defn number-of-hidden-neurons
  ; Returns the number of hidden neurons in NeuralNet net
  [net]
  (- (ncol (.hidden-weights net)) 1))

(defn number-of-output-neurons
  ; Returns the number of output neurons in NeuralNet net
  [net]
  (ncol (.output-weights net)))

(defn forward-pass-hidden
  [net input-vector]
  (let [ilf (mmult input-vector (.hidden-weights net))
        hlv (matrix (mapv activation-function ilf))]
    (->HiddenLayer input-vector ilf hlv)))

(defn forward-pass-output
  [net hl]
  (let [ilf (mmult (trans (.hidden-layer-values hl)) (.output-weights net))
        olv (matrix (mapv activation-function ilf))]
    (->OutputLayer hl ilf olv)))

(defn forward-pass
  [net input-vector]
  (let [hl (forward-pass-hidden net input-vector)
        ol (forward-pass-output net hl)]
    (->ForwardPassResults hl ol)))

(defn evaluate-network
  [net input-vector]
  (let [input-vector-transpose (trans (matrix input-vector))
        forward-pass-results (forward-pass net input-vector-transpose)]
    (.output-layer-values (.output-layer forward-pass-results))))
