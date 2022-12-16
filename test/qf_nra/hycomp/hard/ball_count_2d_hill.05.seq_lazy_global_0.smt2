(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the quantifier free encoding encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:41:51 2012
(declare-fun b.time__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.speed_loss__AT0 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.x__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.speed_x__AT0 () Real)
(assert (let ((.def_77 (* (- 1.0) b.speed_x__AT0)))
(let ((.def_78 (* b.speed_x__AT0 .def_77)))
(let ((.def_79 (* (- 1.0) .def_78)))
(let ((.def_113 (* 2.0 .def_79)))
(let ((.def_133 (* (- 1.0) .def_113)))
(let ((.def_134 (* b.delta__AT0 .def_133)))
(let ((.def_135 (* (- 5.0) .def_134)))
(let ((.def_59 (* (- 1.0) b.x__AT0)))
(let ((.def_92 (* b.speed_x__AT0 .def_59)))
(let ((.def_120 (* (- 5.0) .def_92)))
(let ((.def_140 (+ .def_120 .def_135)))
(let ((.def_87 (* b.x__AT0 .def_77)))
(let ((.def_118 (* (- 5.0) .def_87)))
(let ((.def_141 (+ .def_118 .def_140)))
(let ((.def_122 (* (- 49.0) b.delta__AT0)))
(let ((.def_142 (+ .def_122 .def_141)))
(let ((.def_124 (* 5.0 b.speed_y__AT0)))
(let ((.def_143 (+ .def_124 .def_142)))
(let ((.def_144 (<= .def_143 0.0 )))
(let ((.def_93 (* (- 1.0) .def_92)))
(let ((.def_88 (* (- 1.0) .def_87)))
(let ((.def_110 (+ .def_88 .def_93)))
(let ((.def_111 (+ b.speed_y__AT0 .def_110)))
(let ((.def_132 (<= .def_111 0.0 )))
(let ((.def_145 (and .def_132 .def_144)))
(let ((.def_114 (* b.delta__AT0 .def_113)))
(let ((.def_116 (* 5.0 .def_114)))
(let ((.def_126 (+ .def_120 .def_116)))
(let ((.def_127 (+ .def_118 .def_126)))
(let ((.def_128 (+ .def_122 .def_127)))
(let ((.def_129 (+ .def_124 .def_128)))
(let ((.def_130 (<= 0.0 .def_129)))
(let ((.def_112 (<= 0.0 .def_111)))
(let ((.def_131 (and .def_112 .def_130)))
(let ((.def_146 (or .def_131 .def_145)))
(let ((.def_94 (* b.delta__AT0 .def_93)))
(let ((.def_95 (* 10.0 .def_94)))
(let ((.def_89 (* b.delta__AT0 .def_88)))
(let ((.def_90 (* 10.0 .def_89)))
(let ((.def_102 (+ .def_90 .def_95)))
(let ((.def_76 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_80 (* .def_76 .def_79)))
(let ((.def_81 (* 10.0 .def_80)))
(let ((.def_103 (+ .def_81 .def_102)))
(let ((.def_85 (* (- 49.0) .def_76)))
(let ((.def_104 (+ .def_85 .def_103)))
(let ((.def_74 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_75 (* 10.0 .def_74)))
(let ((.def_105 (+ .def_75 .def_104)))
(let ((.def_60 (* b.x__AT0 .def_59)))
(let ((.def_98 (* (- 10.0) .def_60)))
(let ((.def_106 (+ .def_98 .def_105)))
(let ((.def_100 (* 10.0 b.y__AT0)))
(let ((.def_107 (+ .def_100 .def_106)))
(let ((.def_108 (<= 0.0 .def_107)))
(let ((.def_61 (<= .def_60 b.y__AT0)))
(let ((.def_109 (and .def_61 .def_108)))
(let ((.def_147 (and .def_109 .def_146)))
(let ((.def_67 (<= b.speed_loss__AT0 (/ 1 2))))
(let ((.def_64 (<= (/ 1 10) b.speed_loss__AT0)))
(let ((.def_68 (and .def_64 .def_67)))
(let ((.def_148 (and .def_68 .def_147)))
(let ((.def_69 (and .def_61 .def_68)))
(let ((.def_55 (not b.EVENT.0__AT0)))
(let ((.def_53 (not b.EVENT.1__AT0)))
(let ((.def_56 (or .def_53 .def_55)))
(let ((.def_9 (not b.counter.0__AT0)))
(let ((.def_6 (not b.counter.1__AT0)))
(let ((.def_46 (or .def_6 .def_9)))
(let ((.def_50 (or b.counter.3__AT0 .def_46)))
(let ((.def_47 (or b.counter.2__AT0 .def_46)))
(let ((.def_4 (not b.counter.2__AT0)))
(let ((.def_45 (or .def_4 .def_9)))
(let ((.def_48 (and .def_45 .def_47)))
(let ((.def_14 (not b.counter.3__AT0)))
(let ((.def_49 (or .def_14 .def_48)))
(let ((.def_51 (and .def_49 .def_50)))
(let ((.def_57 (and .def_51 .def_56)))
(let ((.def_70 (and .def_57 .def_69)))
(let ((.def_10 (and .def_6 .def_9)))
(let ((.def_42 (and .def_4 .def_10)))
(let ((.def_43 (and .def_14 .def_42)))
(let ((.def_40 (= b.speed_y__AT0 1.0 )))
(let ((.def_37 (= b.speed_x__AT0 1.0 )))
(let ((.def_33 (= b.y__AT0 10.0 )))
(let ((.def_29 (= b.x__AT0 0.0 )))
(let ((.def_25 (= b.time__AT0 0.0 )))
(let ((.def_27 (and .def_25 b.event_is_timed__AT0)))
(let ((.def_30 (and .def_27 .def_29)))
(let ((.def_34 (and .def_30 .def_33)))
(let ((.def_38 (and .def_34 .def_37)))
(let ((.def_41 (and .def_38 .def_40)))
(let ((.def_44 (and .def_41 .def_43)))
(let ((.def_71 (and .def_44 .def_70)))
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
(let ((.def_72 (and .def_22 .def_71)))
(let ((.def_149 (and .def_72 .def_148)))
.def_149))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
