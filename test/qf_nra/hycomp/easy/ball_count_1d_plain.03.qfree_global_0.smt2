(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using qepcad encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:05 2012
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.0__AT0 () Bool)
(assert (let ((.def_65 (* 49.0 b.delta__AT0)))
(let ((.def_66 (* b.delta__AT0 .def_65)))
(let ((.def_68 (* (- 1.0) .def_66)))
(let ((.def_62 (* 10.0 b.delta__AT0)))
(let ((.def_63 (* b.speed_y__AT0 .def_62)))
(let ((.def_69 (+ .def_63 .def_68)))
(let ((.def_70 (* 10.0 b.y__AT0)))
(let ((.def_72 (+ .def_70 .def_69)))
(let ((.def_73 (<= 0.0 .def_72)))
(let ((.def_52 (<= 0.0 b.y__AT0)))
(let ((.def_74 (and .def_52 .def_73)))
(let ((.def_60 (<= 0.0 b.delta__AT0)))
(let ((.def_61 (not .def_60)))
(let ((.def_75 (or .def_61 .def_74)))
(let ((.def_41 (not b.EVENT.0__AT0)))
(let ((.def_39 (not b.EVENT.1__AT0)))
(let ((.def_57 (and .def_39 .def_41)))
(let ((.def_58 (not .def_57)))
(let ((.def_76 (or .def_58 .def_75)))
(let ((.def_50 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_47 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_51 (and .def_47 .def_50)))
(let ((.def_53 (and .def_51 .def_52)))
(let ((.def_42 (or .def_39 .def_41)))
(let ((.def_6 (not b.counter.0__AT0)))
(let ((.def_4 (not b.counter.1__AT0)))
(let ((.def_32 (or .def_4 .def_6)))
(let ((.def_36 (or b.counter.3__AT0 .def_32)))
(let ((.def_33 (or b.counter.2__AT0 .def_32)))
(let ((.def_9 (not b.counter.2__AT0)))
(let ((.def_31 (or .def_6 .def_9)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_12 (not b.counter.3__AT0)))
(let ((.def_35 (or .def_12 .def_34)))
(let ((.def_37 (and .def_35 .def_36)))
(let ((.def_43 (and .def_37 .def_42)))
(let ((.def_54 (and .def_43 .def_53)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_29 (and .def_10 .def_12)))
(let ((.def_27 (= b.speed_y__AT0 0.0 )))
(let ((.def_24 (= b.y__AT0 10.0 )))
(let ((.def_19 (= b.time__AT0 0.0 )))
(let ((.def_21 (and .def_19 b.event_is_timed__AT0)))
(let ((.def_25 (and .def_21 .def_24)))
(let ((.def_28 (and .def_25 .def_27)))
(let ((.def_30 (and .def_28 .def_29)))
(let ((.def_55 (and .def_30 .def_54)))
(let ((.def_14 (or .def_7 b.counter.3__AT0)))
(let ((.def_13 (or .def_10 .def_12)))
(let ((.def_15 (and .def_13 .def_14)))
(let ((.def_16 (not .def_15)))
(let ((.def_56 (and .def_16 .def_55)))
(let ((.def_77 (and .def_56 .def_76)))
.def_77)))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)