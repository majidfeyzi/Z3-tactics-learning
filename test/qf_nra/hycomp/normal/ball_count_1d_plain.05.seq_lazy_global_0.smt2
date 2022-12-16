(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the quantifier free encoding encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:31 2012
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(assert (let ((.def_78 (* (- 49.0) b.delta__AT0)))
(let ((.def_80 (* 5.0 b.speed_y__AT0)))
(let ((.def_82 (+ .def_80 .def_78)))
(let ((.def_86 (<= .def_82 0.0 )))
(let ((.def_85 (<= b.speed_y__AT0 0.0 )))
(let ((.def_87 (and .def_85 .def_86)))
(let ((.def_83 (<= 0.0 .def_82)))
(let ((.def_77 (<= 0.0 b.speed_y__AT0)))
(let ((.def_84 (and .def_77 .def_83)))
(let ((.def_88 (or .def_84 .def_87)))
(let ((.def_69 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_70 (* 10.0 .def_69)))
(let ((.def_65 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_68 (* (- 49.0) .def_65)))
(let ((.def_71 (+ .def_68 .def_70)))
(let ((.def_72 (* 10.0 b.y__AT0)))
(let ((.def_74 (+ .def_72 .def_71)))
(let ((.def_75 (<= 0.0 .def_74)))
(let ((.def_51 (<= 0.0 b.y__AT0)))
(let ((.def_76 (and .def_51 .def_75)))
(let ((.def_89 (and .def_76 .def_88)))
(let ((.def_58 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_55 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_59 (and .def_55 .def_58)))
(let ((.def_90 (and .def_59 .def_89)))
(let ((.def_60 (and .def_51 .def_59)))
(let ((.def_48 (not b.EVENT.0__AT0)))
(let ((.def_46 (not b.EVENT.1__AT0)))
(let ((.def_49 (or .def_46 .def_48)))
(let ((.def_9 (not b.counter.0__AT0)))
(let ((.def_6 (not b.counter.1__AT0)))
(let ((.def_39 (or .def_6 .def_9)))
(let ((.def_43 (or b.counter.3__AT0 .def_39)))
(let ((.def_40 (or b.counter.2__AT0 .def_39)))
(let ((.def_4 (not b.counter.2__AT0)))
(let ((.def_38 (or .def_4 .def_9)))
(let ((.def_41 (and .def_38 .def_40)))
(let ((.def_14 (not b.counter.3__AT0)))
(let ((.def_42 (or .def_14 .def_41)))
(let ((.def_44 (and .def_42 .def_43)))
(let ((.def_50 (and .def_44 .def_49)))
(let ((.def_61 (and .def_50 .def_60)))
(let ((.def_10 (and .def_6 .def_9)))
(let ((.def_35 (and .def_4 .def_10)))
(let ((.def_36 (and .def_14 .def_35)))
(let ((.def_33 (= b.speed_y__AT0 0.0 )))
(let ((.def_30 (= b.y__AT0 10.0 )))
(let ((.def_25 (= b.time__AT0 0.0 )))
(let ((.def_27 (and .def_25 b.event_is_timed__AT0)))
(let ((.def_31 (and .def_27 .def_30)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_37 (and .def_34 .def_36)))
(let ((.def_62 (and .def_37 .def_61)))
(let ((.def_17 (or b.counter.1__AT0 .def_9)))
(let ((.def_18 (or b.counter.2__AT0 .def_17)))
(let ((.def_16 (or .def_4 .def_10)))
(let ((.def_19 (and .def_16 .def_18)))
(let ((.def_20 (or b.counter.3__AT0 .def_19)))
(let ((.def_11 (or b.counter.2__AT0 .def_10)))
(let ((.def_7 (or .def_4 .def_6)))
(let ((.def_12 (and .def_7 .def_11)))
(let ((.def_15 (or .def_12 .def_14)))
(let ((.def_21 (and .def_15 .def_20)))
(let ((.def_22 (not .def_21)))
(let ((.def_63 (and .def_22 .def_62)))
(let ((.def_91 (and .def_63 .def_90)))
.def_91)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
