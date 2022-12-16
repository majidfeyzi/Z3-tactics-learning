(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using redlog encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:04 2012
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.x__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.g__AT0 () Real)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(assert (let ((.def_77 (* b.x__AT0 b.x__AT0)))
(let ((.def_73 (* 5.0 b.delta__AT0)))
(let ((.def_74 (* b.delta__AT0 .def_73)))
(let ((.def_75 (* (- 1.0) .def_74)))
(let ((.def_80 (+ .def_75 .def_77)))
(let ((.def_71 (* b.delta__AT0 b.speed_y__AT0)))
(let ((.def_81 (+ .def_71 .def_80)))
(let ((.def_82 (+ b.y__AT0 .def_81)))
(let ((.def_83 (<= 0.0 .def_82)))
(let ((.def_119 (not .def_83)))
(let ((.def_100 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_101 (* 10.0 b.delta__AT0)))
(let ((.def_102 (+ .def_101 .def_100)))
(let ((.def_103 (<= .def_102 0.0 )))
(let ((.def_120 (and .def_103 .def_119)))
(let ((.def_112 (+ b.y__AT0 .def_77)))
(let ((.def_117 (<= 0.0 .def_112)))
(let ((.def_114 (<= b.speed_y__AT0 0.0 )))
(let ((.def_118 (or .def_114 .def_117)))
(let ((.def_121 (or .def_118 .def_120)))
(let ((.def_113 (<= .def_112 0.0 )))
(let ((.def_115 (and .def_113 .def_114)))
(let ((.def_107 (<= 0.0 .def_102)))
(let ((.def_108 (not .def_107)))
(let ((.def_105 (= .def_82 0.0 )))
(let ((.def_106 (not .def_105)))
(let ((.def_109 (or .def_106 .def_108)))
(let ((.def_104 (or .def_83 .def_103)))
(let ((.def_110 (and .def_104 .def_109)))
(let ((.def_87 (* 20.0 b.x__AT0)))
(let ((.def_88 (* b.x__AT0 .def_87)))
(let ((.def_85 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_89 (+ .def_85 .def_88)))
(let ((.def_90 (* 20.0 b.y__AT0)))
(let ((.def_92 (+ .def_90 .def_89)))
(let ((.def_95 (<= .def_92 0.0 )))
(let ((.def_96 (not .def_95)))
(let ((.def_111 (or .def_96 .def_110)))
(let ((.def_116 (or .def_111 .def_115)))
(let ((.def_122 (and .def_116 .def_121)))
(let ((.def_93 (<= 0.0 .def_92)))
(let ((.def_94 (not .def_93)))
(let ((.def_123 (or .def_94 .def_122)))
(let ((.def_68 (<= 0.0 b.delta__AT0)))
(let ((.def_69 (not .def_68)))
(let ((.def_84 (or .def_69 .def_83)))
(let ((.def_124 (and .def_84 .def_123)))
(let ((.def_48 (not b.EVENT.0__AT0)))
(let ((.def_46 (not b.EVENT.1__AT0)))
(let ((.def_65 (and .def_46 .def_48)))
(let ((.def_66 (not .def_65)))
(let ((.def_125 (or .def_66 .def_124)))
(let ((.def_59 (<= b.g__AT0 10.0 )))
(let ((.def_58 (<= 8.0 b.g__AT0)))
(let ((.def_60 (and .def_58 .def_59)))
(let ((.def_53 (* (- 1.0) b.x__AT0)))
(let ((.def_54 (* b.x__AT0 .def_53)))
(let ((.def_55 (<= .def_54 b.y__AT0)))
(let ((.def_61 (and .def_55 .def_60)))
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
(let ((.def_62 (and .def_50 .def_61)))
(let ((.def_10 (and .def_6 .def_9)))
(let ((.def_35 (and .def_4 .def_10)))
(let ((.def_36 (and .def_14 .def_35)))
(let ((.def_33 (= b.y__AT0 10.0 )))
(let ((.def_29 (= b.x__AT0 0.0 )))
(let ((.def_25 (= b.time__AT0 0.0 )))
(let ((.def_27 (and .def_25 b.event_is_timed__AT0)))
(let ((.def_30 (and .def_27 .def_29)))
(let ((.def_34 (and .def_30 .def_33)))
(let ((.def_37 (and .def_34 .def_36)))
(let ((.def_63 (and .def_37 .def_62)))
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
(let ((.def_64 (and .def_22 .def_63)))
(let ((.def_126 (and .def_64 .def_125)))
.def_126))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)