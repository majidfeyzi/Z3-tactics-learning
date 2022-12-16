(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using redlog encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:40:14 2012
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
(assert (let ((.def_70 (* b.x__AT0 b.x__AT0)))
(let ((.def_66 (* 5.0 b.delta__AT0)))
(let ((.def_67 (* b.delta__AT0 .def_66)))
(let ((.def_68 (* (- 1.0) .def_67)))
(let ((.def_73 (+ .def_68 .def_70)))
(let ((.def_64 (* b.delta__AT0 b.speed_y__AT0)))
(let ((.def_74 (+ .def_64 .def_73)))
(let ((.def_75 (+ b.y__AT0 .def_74)))
(let ((.def_76 (<= 0.0 .def_75)))
(let ((.def_112 (not .def_76)))
(let ((.def_93 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_94 (* 10.0 b.delta__AT0)))
(let ((.def_95 (+ .def_94 .def_93)))
(let ((.def_96 (<= .def_95 0.0 )))
(let ((.def_113 (and .def_96 .def_112)))
(let ((.def_105 (+ b.y__AT0 .def_70)))
(let ((.def_110 (<= 0.0 .def_105)))
(let ((.def_107 (<= b.speed_y__AT0 0.0 )))
(let ((.def_111 (or .def_107 .def_110)))
(let ((.def_114 (or .def_111 .def_113)))
(let ((.def_106 (<= .def_105 0.0 )))
(let ((.def_108 (and .def_106 .def_107)))
(let ((.def_100 (<= 0.0 .def_95)))
(let ((.def_101 (not .def_100)))
(let ((.def_98 (= .def_75 0.0 )))
(let ((.def_99 (not .def_98)))
(let ((.def_102 (or .def_99 .def_101)))
(let ((.def_97 (or .def_76 .def_96)))
(let ((.def_103 (and .def_97 .def_102)))
(let ((.def_80 (* 20.0 b.x__AT0)))
(let ((.def_81 (* b.x__AT0 .def_80)))
(let ((.def_78 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_82 (+ .def_78 .def_81)))
(let ((.def_83 (* 20.0 b.y__AT0)))
(let ((.def_85 (+ .def_83 .def_82)))
(let ((.def_88 (<= .def_85 0.0 )))
(let ((.def_89 (not .def_88)))
(let ((.def_104 (or .def_89 .def_103)))
(let ((.def_109 (or .def_104 .def_108)))
(let ((.def_115 (and .def_109 .def_114)))
(let ((.def_86 (<= 0.0 .def_85)))
(let ((.def_87 (not .def_86)))
(let ((.def_116 (or .def_87 .def_115)))
(let ((.def_61 (<= 0.0 b.delta__AT0)))
(let ((.def_62 (not .def_61)))
(let ((.def_77 (or .def_62 .def_76)))
(let ((.def_117 (and .def_77 .def_116)))
(let ((.def_41 (not b.EVENT.0__AT0)))
(let ((.def_39 (not b.EVENT.1__AT0)))
(let ((.def_58 (and .def_39 .def_41)))
(let ((.def_59 (not .def_58)))
(let ((.def_118 (or .def_59 .def_117)))
(let ((.def_52 (<= b.g__AT0 10.0 )))
(let ((.def_51 (<= 8.0 b.g__AT0)))
(let ((.def_53 (and .def_51 .def_52)))
(let ((.def_46 (* (- 1.0) b.x__AT0)))
(let ((.def_47 (* b.x__AT0 .def_46)))
(let ((.def_48 (<= .def_47 b.y__AT0)))
(let ((.def_54 (and .def_48 .def_53)))
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
(let ((.def_55 (and .def_43 .def_54)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_29 (and .def_10 .def_12)))
(let ((.def_27 (= b.y__AT0 10.0 )))
(let ((.def_23 (= b.x__AT0 0.0 )))
(let ((.def_19 (= b.time__AT0 0.0 )))
(let ((.def_21 (and .def_19 b.event_is_timed__AT0)))
(let ((.def_24 (and .def_21 .def_23)))
(let ((.def_28 (and .def_24 .def_27)))
(let ((.def_30 (and .def_28 .def_29)))
(let ((.def_56 (and .def_30 .def_55)))
(let ((.def_14 (or .def_7 b.counter.3__AT0)))
(let ((.def_13 (or .def_10 .def_12)))
(let ((.def_15 (and .def_13 .def_14)))
(let ((.def_16 (not .def_15)))
(let ((.def_57 (and .def_16 .def_56)))
(let ((.def_119 (and .def_57 .def_118)))
.def_119)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
