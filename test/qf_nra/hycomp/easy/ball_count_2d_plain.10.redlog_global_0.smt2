(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using redlog encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:48 2012
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun g__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.y__AT1 () Real)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(assert (let ((.def_160 (= b.y__AT0 b.y__AT1)))
(let ((.def_161 (not .def_160)))
(let ((.def_66 (* g__AT0 b.delta__AT0)))
(let ((.def_148 (<= b.speed_y__AT0 .def_66)))
(let ((.def_149 (not .def_148)))
(let ((.def_70 (* 2.0 b.speed_y__AT0)))
(let ((.def_71 (* b.delta__AT0 .def_70)))
(let ((.def_67 (* b.delta__AT0 .def_66)))
(let ((.def_69 (* (- 1.0) .def_67)))
(let ((.def_72 (+ .def_69 .def_71)))
(let ((.def_73 (* 2.0 b.y__AT0)))
(let ((.def_75 (+ .def_73 .def_72)))
(let ((.def_140 (= .def_75 0.0 )))
(let ((.def_141 (not .def_140)))
(let ((.def_150 (or .def_141 .def_149)))
(let ((.def_103 (* 2.0 g__AT0)))
(let ((.def_127 (* b.speed_y__AT0 .def_103)))
(let ((.def_128 (* b.delta__AT0 .def_127)))
(let ((.def_123 (* g__AT0 g__AT0)))
(let ((.def_124 (* b.delta__AT0 .def_123)))
(let ((.def_125 (* b.delta__AT0 .def_124)))
(let ((.def_126 (* (- 1.0) .def_125)))
(let ((.def_129 (+ .def_126 .def_128)))
(let ((.def_104 (* b.y__AT0 .def_103)))
(let ((.def_130 (+ .def_104 .def_129)))
(let ((.def_135 (<= 0.0 .def_130)))
(let ((.def_80 (* b.speed_y__AT0 g__AT0)))
(let ((.def_132 (<= .def_124 .def_80)))
(let ((.def_136 (or .def_132 .def_135)))
(let ((.def_151 (and .def_136 .def_150)))
(let ((.def_82 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_105 (+ .def_82 .def_104)))
(let ((.def_145 (<= .def_105 0.0 )))
(let ((.def_146 (not .def_145)))
(let ((.def_119 (<= 0.0 g__AT0)))
(let ((.def_120 (not .def_119)))
(let ((.def_147 (or .def_120 .def_146)))
(let ((.def_152 (or .def_147 .def_151)))
(let ((.def_109 (* b.y__AT0 g__AT0)))
(let ((.def_115 (<= .def_109 0.0 )))
(let ((.def_114 (<= .def_80 0.0 )))
(let ((.def_116 (and .def_114 .def_115)))
(let ((.def_153 (or .def_116 .def_152)))
(let ((.def_138 (<= .def_66 b.speed_y__AT0)))
(let ((.def_139 (not .def_138)))
(let ((.def_142 (or .def_139 .def_141)))
(let ((.def_131 (<= .def_130 0.0 )))
(let ((.def_133 (and .def_131 .def_132)))
(let ((.def_134 (or .def_120 .def_133)))
(let ((.def_137 (and .def_134 .def_136)))
(let ((.def_143 (and .def_137 .def_142)))
(let ((.def_118 (= .def_105 0.0 )))
(let ((.def_121 (and .def_118 .def_120)))
(let ((.def_111 (<= g__AT0 0.0 )))
(let ((.def_112 (not .def_111)))
(let ((.def_110 (<= 0.0 .def_109)))
(let ((.def_113 (and .def_110 .def_112)))
(let ((.def_117 (or .def_113 .def_116)))
(let ((.def_122 (or .def_117 .def_121)))
(let ((.def_144 (or .def_122 .def_143)))
(let ((.def_154 (and .def_144 .def_153)))
(let ((.def_106 (<= 0.0 .def_105)))
(let ((.def_107 (not .def_106)))
(let ((.def_86 (= g__AT0 0.0 )))
(let ((.def_108 (or .def_86 .def_107)))
(let ((.def_155 (or .def_108 .def_154)))
(let ((.def_96 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_97 (+ b.y__AT0 .def_96)))
(let ((.def_98 (= .def_97 0.0 )))
(let ((.def_99 (not .def_98)))
(let ((.def_93 (* b.delta__AT0 .def_82)))
(let ((.def_89 (* b.y__AT0 b.speed_y__AT0)))
(let ((.def_94 (+ .def_89 .def_93)))
(let ((.def_95 (<= .def_94 0.0 )))
(let ((.def_100 (and .def_95 .def_99)))
(let ((.def_90 (<= 0.0 .def_89)))
(let ((.def_87 (not .def_86)))
(let ((.def_83 (* b.speed_y__AT0 .def_82)))
(let ((.def_81 (* b.y__AT0 .def_80)))
(let ((.def_84 (+ .def_81 .def_83)))
(let ((.def_85 (<= .def_84 0.0 )))
(let ((.def_88 (or .def_85 .def_87)))
(let ((.def_91 (or .def_88 .def_90)))
(let ((.def_23 (= b.speed_y__AT0 0.0 )))
(let ((.def_92 (or .def_23 .def_91)))
(let ((.def_101 (or .def_92 .def_100)))
(let ((.def_77 (<= 0.0 b.delta__AT0)))
(let ((.def_78 (not .def_77)))
(let ((.def_76 (<= 0.0 .def_75)))
(let ((.def_79 (or .def_76 .def_78)))
(let ((.def_102 (and .def_79 .def_101)))
(let ((.def_156 (and .def_102 .def_155)))
(let ((.def_41 (not b.EVENT.0__AT0)))
(let ((.def_39 (not b.EVENT.1__AT0)))
(let ((.def_63 (and .def_39 .def_41)))
(let ((.def_157 (and .def_63 .def_156)))
(let ((.def_158 (not .def_157)))
(let ((.def_162 (or .def_158 .def_161)))
(let ((.def_64 (not .def_63)))
(let ((.def_163 (or .def_64 .def_162)))
(let ((.def_58 (<= 0.0 b.y__AT0)))
(let ((.def_55 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_52 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_56 (and .def_52 .def_55)))
(let ((.def_47 (<= g__AT0 10.0 )))
(let ((.def_46 (<= 8.0 g__AT0)))
(let ((.def_48 (and .def_46 .def_47)))
(let ((.def_57 (and .def_48 .def_56)))
(let ((.def_59 (and .def_57 .def_58)))
(let ((.def_42 (or .def_39 .def_41)))
(let ((.def_25 (not b.counter.1__AT0)))
(let ((.def_4 (not b.counter.0__AT0)))
(let ((.def_32 (or .def_4 .def_25)))
(let ((.def_36 (or b.counter.3__AT0 .def_32)))
(let ((.def_33 (or b.counter.2__AT0 .def_32)))
(let ((.def_8 (not b.counter.2__AT0)))
(let ((.def_31 (or .def_4 .def_8)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_28 (not b.counter.3__AT0)))
(let ((.def_35 (or .def_28 .def_34)))
(let ((.def_37 (and .def_35 .def_36)))
(let ((.def_43 (and .def_37 .def_42)))
(let ((.def_60 (and .def_43 .def_59)))
(let ((.def_26 (and .def_4 .def_25)))
(let ((.def_27 (and .def_8 .def_26)))
(let ((.def_29 (and .def_27 .def_28)))
(let ((.def_20 (= b.y__AT0 10.0 )))
(let ((.def_15 (= b.time__AT0 0.0 )))
(let ((.def_17 (and .def_15 b.event_is_timed__AT0)))
(let ((.def_21 (and .def_17 .def_20)))
(let ((.def_24 (and .def_21 .def_23)))
(let ((.def_30 (and .def_24 .def_29)))
(let ((.def_61 (and .def_30 .def_60)))
(let ((.def_6 (or .def_4 b.counter.1__AT0)))
(let ((.def_9 (or .def_6 .def_8)))
(let ((.def_11 (or .def_9 b.counter.3__AT0)))
(let ((.def_12 (not .def_11)))
(let ((.def_62 (and .def_12 .def_61)))
(let ((.def_164 (and .def_62 .def_163)))
.def_164))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)