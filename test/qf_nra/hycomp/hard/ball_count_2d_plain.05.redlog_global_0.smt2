(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using redlog encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:45:53 2012
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
(assert (let ((.def_167 (= b.y__AT0 b.y__AT1)))
(let ((.def_168 (not .def_167)))
(let ((.def_73 (* g__AT0 b.delta__AT0)))
(let ((.def_155 (<= b.speed_y__AT0 .def_73)))
(let ((.def_156 (not .def_155)))
(let ((.def_77 (* 2.0 b.speed_y__AT0)))
(let ((.def_78 (* b.delta__AT0 .def_77)))
(let ((.def_74 (* b.delta__AT0 .def_73)))
(let ((.def_76 (* (- 1.0) .def_74)))
(let ((.def_79 (+ .def_76 .def_78)))
(let ((.def_80 (* 2.0 b.y__AT0)))
(let ((.def_82 (+ .def_80 .def_79)))
(let ((.def_147 (= .def_82 0.0 )))
(let ((.def_148 (not .def_147)))
(let ((.def_157 (or .def_148 .def_156)))
(let ((.def_110 (* 2.0 g__AT0)))
(let ((.def_134 (* b.speed_y__AT0 .def_110)))
(let ((.def_135 (* b.delta__AT0 .def_134)))
(let ((.def_130 (* g__AT0 g__AT0)))
(let ((.def_131 (* b.delta__AT0 .def_130)))
(let ((.def_132 (* b.delta__AT0 .def_131)))
(let ((.def_133 (* (- 1.0) .def_132)))
(let ((.def_136 (+ .def_133 .def_135)))
(let ((.def_111 (* b.y__AT0 .def_110)))
(let ((.def_137 (+ .def_111 .def_136)))
(let ((.def_142 (<= 0.0 .def_137)))
(let ((.def_87 (* b.speed_y__AT0 g__AT0)))
(let ((.def_139 (<= .def_131 .def_87)))
(let ((.def_143 (or .def_139 .def_142)))
(let ((.def_158 (and .def_143 .def_157)))
(let ((.def_89 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_112 (+ .def_89 .def_111)))
(let ((.def_152 (<= .def_112 0.0 )))
(let ((.def_153 (not .def_152)))
(let ((.def_126 (<= 0.0 g__AT0)))
(let ((.def_127 (not .def_126)))
(let ((.def_154 (or .def_127 .def_153)))
(let ((.def_159 (or .def_154 .def_158)))
(let ((.def_116 (* b.y__AT0 g__AT0)))
(let ((.def_122 (<= .def_116 0.0 )))
(let ((.def_121 (<= .def_87 0.0 )))
(let ((.def_123 (and .def_121 .def_122)))
(let ((.def_160 (or .def_123 .def_159)))
(let ((.def_145 (<= .def_73 b.speed_y__AT0)))
(let ((.def_146 (not .def_145)))
(let ((.def_149 (or .def_146 .def_148)))
(let ((.def_138 (<= .def_137 0.0 )))
(let ((.def_140 (and .def_138 .def_139)))
(let ((.def_141 (or .def_127 .def_140)))
(let ((.def_144 (and .def_141 .def_143)))
(let ((.def_150 (and .def_144 .def_149)))
(let ((.def_125 (= .def_112 0.0 )))
(let ((.def_128 (and .def_125 .def_127)))
(let ((.def_118 (<= g__AT0 0.0 )))
(let ((.def_119 (not .def_118)))
(let ((.def_117 (<= 0.0 .def_116)))
(let ((.def_120 (and .def_117 .def_119)))
(let ((.def_124 (or .def_120 .def_123)))
(let ((.def_129 (or .def_124 .def_128)))
(let ((.def_151 (or .def_129 .def_150)))
(let ((.def_161 (and .def_151 .def_160)))
(let ((.def_113 (<= 0.0 .def_112)))
(let ((.def_114 (not .def_113)))
(let ((.def_93 (= g__AT0 0.0 )))
(let ((.def_115 (or .def_93 .def_114)))
(let ((.def_162 (or .def_115 .def_161)))
(let ((.def_103 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_104 (+ b.y__AT0 .def_103)))
(let ((.def_105 (= .def_104 0.0 )))
(let ((.def_106 (not .def_105)))
(let ((.def_100 (* b.delta__AT0 .def_89)))
(let ((.def_96 (* b.y__AT0 b.speed_y__AT0)))
(let ((.def_101 (+ .def_96 .def_100)))
(let ((.def_102 (<= .def_101 0.0 )))
(let ((.def_107 (and .def_102 .def_106)))
(let ((.def_97 (<= 0.0 .def_96)))
(let ((.def_94 (not .def_93)))
(let ((.def_90 (* b.speed_y__AT0 .def_89)))
(let ((.def_88 (* b.y__AT0 .def_87)))
(let ((.def_91 (+ .def_88 .def_90)))
(let ((.def_92 (<= .def_91 0.0 )))
(let ((.def_95 (or .def_92 .def_94)))
(let ((.def_98 (or .def_95 .def_97)))
(let ((.def_33 (= b.speed_y__AT0 0.0 )))
(let ((.def_99 (or .def_33 .def_98)))
(let ((.def_108 (or .def_99 .def_107)))
(let ((.def_84 (<= 0.0 b.delta__AT0)))
(let ((.def_85 (not .def_84)))
(let ((.def_83 (<= 0.0 .def_82)))
(let ((.def_86 (or .def_83 .def_85)))
(let ((.def_109 (and .def_86 .def_108)))
(let ((.def_163 (and .def_109 .def_162)))
(let ((.def_48 (not b.EVENT.0__AT0)))
(let ((.def_46 (not b.EVENT.1__AT0)))
(let ((.def_70 (and .def_46 .def_48)))
(let ((.def_164 (and .def_70 .def_163)))
(let ((.def_165 (not .def_164)))
(let ((.def_169 (or .def_165 .def_168)))
(let ((.def_71 (not .def_70)))
(let ((.def_170 (or .def_71 .def_169)))
(let ((.def_65 (<= 0.0 b.y__AT0)))
(let ((.def_62 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_59 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_63 (and .def_59 .def_62)))
(let ((.def_54 (<= g__AT0 10.0 )))
(let ((.def_53 (<= 8.0 g__AT0)))
(let ((.def_55 (and .def_53 .def_54)))
(let ((.def_64 (and .def_55 .def_63)))
(let ((.def_66 (and .def_64 .def_65)))
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
(let ((.def_67 (and .def_50 .def_66)))
(let ((.def_10 (and .def_6 .def_9)))
(let ((.def_35 (and .def_4 .def_10)))
(let ((.def_36 (and .def_14 .def_35)))
(let ((.def_30 (= b.y__AT0 10.0 )))
(let ((.def_25 (= b.time__AT0 0.0 )))
(let ((.def_27 (and .def_25 b.event_is_timed__AT0)))
(let ((.def_31 (and .def_27 .def_30)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_37 (and .def_34 .def_36)))
(let ((.def_68 (and .def_37 .def_67)))
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
(let ((.def_69 (and .def_22 .def_68)))
(let ((.def_171 (and .def_69 .def_170)))
.def_171)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
