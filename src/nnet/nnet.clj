(ns nnet.nnet
  (:require [nnet.data-structures :refer :all]
            [clojure.core.matrix :as m])
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
  (- (m/row-count (.hidden-weights net)) 1))

(defn number-of-hidden-neurons
  ; Returns the number of hidden neurons in NeuralNet net
  [net]
  (- (m/column-count (.hidden-weights net)) 1))

(defn number-of-output-neurons
  ; Returns the number of output neurons in NeuralNet net
  [net]
  (m/column-count (.output-weights net)))

(def activ-func-mapper (partial mapv activation-function))

(defn forward-pass-hidden
  [net input-vector]
  (let [ilf (m/mmul input-vector (.hidden-weights net))
        hlv (mapv activ-func-mapper ilf)]
    (->HiddenLayer input-vector ilf hlv)))

(defn forward-pass-output
  [net hl]
  ; the problem is that the new transpose function doesn't transpose 1-d vectors. Can't imagine why
  ; they made it that way....
  (let [ilf (m/mmul (m/transpose (.hidden-layer-values hl)) (.output-weights net))
        olv (mapv activ-func-mapper ilf)]
    (->OutputLayer hl ilf olv)))

(defn forward-pass
  [net input-vector]
  (let [hl (forward-pass-hidden net input-vector)
        ol (forward-pass-output net hl)]
    (->ForwardPassResults hl ol)))

(defn evaluate-network
  [net input-vector]
  (let [input-vector-transpose (m/transpose input-vector)
        forward-pass-results (forward-pass net input-vector-transpose)]
    (.output-layer-values (.output-layer forward-pass-results))))
