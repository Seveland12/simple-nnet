(ns nnet.core
  (:gen-class)
  (:use [nnet.nnet :as n]
        [nnet.backprop :as backprop]
        [nnet.examples.xor :as xor]
        [nnet.examples.calc-dig :as calc-dig]
        [clojure.core.matrix :as m]))

(def ex0 (xor/training-set-xor 0))
(def i0 (:input-vector ex0))
(def d0 (:desired-response ex0))

(def ex1 (xor/training-set-xor 1))
(def i1 (:input-vector ex1))
(def d1 (:desired-response ex1))

(def ex2 (xor/training-set-xor 2))
(def i2 (:input-vector ex2))
(def d2 (:desired-response ex2))

(def ex3 (xor/training-set-xor 3))
(def i3 (:input-vector ex3))
(def d3 (:desired-response ex3))

(def mynet-xor (backprop/initial-net xor/training-set-xor))

(def mynet-calc-dig (backprop/initial-net calc-dig/training-set-calc-dig))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))










