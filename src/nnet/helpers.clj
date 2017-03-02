(ns nnet.helpers
  (:require [clojure.core.matrix :as m])
  )



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
