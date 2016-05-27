(ns nnet.backprop
  (:require (incanter [core :refer :all]))
  (:use ;; [nnet.nnet :as n :only [activation-function
        ;;                         hidden-layer
        ;;                         output-layer
        ;;                         evaluate-network]]
        [nnet.math-utilities :as utils :only [approx-equals?]])) 

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

(defn error-function
  [err-vector]
  (reduce + (map utils/my-sq err-vector)))

; output weights: del wij = (lambda)(ej)(Phi-prime(vj)(yi)
; vj = induced local field of neuron j (mmult (trans h) w)

(defrecord NeuralNet [hidden-weights output-weights])

(defrecord HiddenLayer [input-values induced-local-field hidden-layer-values])
(defrecord OutputLayer [hidden-layer induced-local-field output-layer-values])
(defrecord ForwardPassResults [hidden-layer output-layer])

(defrecord BackwardPassOL [forward-pass-results error-vector-output del-output delta-W-output])

(def test-net (->NeuralNet my-wh my-wo))

(defn forward-pass-hidden
  [net input-vector]
  (let [ilf (mmult input-vector (.hidden-weights net))]
    (let [hlv (matrix (mapv activation-function ilf))]
      (->HiddenLayer input-vector ilf hlv))))

(defn forward-pass-output
  [net hl]
  (let [ilf (mmult (trans (.hidden-layer-values hl)) (.output-weights net))]
    (let [olv (matrix (mapv activation-function ilf))]
      (->OutputLayer hl ilf olv))))

(defn forward-pass
  [net input-vector]
  (let [hl (forward-pass-hidden net input-vector)]
    (let [ol (forward-pass-output net hl)]
      (->ForwardPassResults hl ol))))

(defn backward-pass-output
  [net desired-response fpr]
  (let [current-error-vector (minus desired-response (.output-layer-values  (.output-layer fpr)))]
    (let [current-del-output (mult current-error-vector (mapv activation-function-deriv (.induced-local-field (.output-layer fpr))))]
      (let [delta-W (mmult (.hidden-layer-values (.hidden-layer fpr)) (trans current-del-output))]
        (->BackwardPassOL fpr current-error-vector current-del-output delta-W)
        ))))

;; (defn train
;;   [net]
;;   (loop []
;;     (let [current-hidden-layer (n/hidden-layer input-vector (.hidden-weights net))]
;;       (let [current-output (n/output-layer current-hidden-layer (.output-weights net))]
;;         (let [current-error-vector (minus desired-response current-output)]
;;           (let [current-error-value (error-function current-error-vector)]
;;             (println current-error-value)
;;             (let [is-minimized? (utils/approx-equals? current-error-value 0.0)]
;;               (println is-minimized?)
;;               (if (not is-minimized?)
                
;;                 (recur)))))))))





