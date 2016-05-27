(ns nnet.backprop
  (:require (incanter [core :refer :all]))
  (:use [nnet.nnet :as n :only [activation-function
                                hidden-layer
                                output-layer
                                evaluate-network]]
        [nnet.math-utilities :as utils :only [approx-equals?]])
  (:import [nnet.nnet NeuralNet])) 

(def my-wh (matrix [[0.362985 0.418378 0.0]
                    [-0.464489 -0.554121 0.0]
                    [-0.720958 0.504430 1.0]]))
(def my-wo (matrix [0.620124 -0.446396 0.692502]))

(def input-vector
  ; input-vector has an extra final component = 1.0 to accomodate the
  ; bias terms
  (trans (matrix [0.5 -0.5 1.0])))

(def desired-response
  (matrix [0.5]))

(def learning-rate 0.001)

(defn activation-function-deriv
  ; Clearly this is the derivative of the activation function.
  ; Hard-coded for now.
  [x]
  (/ 0.1439333 (utils/my-sq (Math/cosh (* 0.66666 x)))))

(defn error-function
  [err-vector]
  (reduce + (map utils/my-sq err-vector)))

; output weights: del wij = (lambda)(ej)(Phi-prime(vj)(yi)
; vj = induced local field of neuron j (mmult (trans h) w)

(defn train
  [n]
  (loop []
    (let [current-hidden-layer (n/hidden-layer input-vector my-wh)]
      (let [current-output (n/output-layer current-hidden-layer my-wo)]
        (let [current-error-vector (minus desired-response current-output)]
          (let [current-error-value (error-function current-error-vector)]
            (println current-error-value)
            (let [is-minimized? (utils/approx-equals? current-error-value 0.0)]
              (println is-minimized?)
              (if (not is-minimized?)
                
                (recur)))))))))




