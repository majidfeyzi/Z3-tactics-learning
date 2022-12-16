(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 1 and uses the quantifier free encoding with lemmas encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:17 2012
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.time__AT1 () Real)
(declare-fun b.y__AT1 () Real)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.delta__AT1 () Real)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.speed_y__AT1 () Real)
(assert (let ((.def_291 (<= b.speed_y__AT1 0.0 )))
(let ((.def_273 (* (- 49.0) b.delta__AT1)))
(let ((.def_274 (* 5.0 b.speed_y__AT1)))
(let ((.def_276 (+ .def_274 .def_273)))
(let ((.def_289 (<= .def_276 0.0 )))
(let ((.def_308 (and .def_289 .def_291)))
(let ((.def_277 (<= 0.0 .def_276)))
(let ((.def_278 (not .def_277)))
(let ((.def_271 (<= 0.0 b.speed_y__AT1)))
(let ((.def_305 (or .def_271 .def_278)))
(let ((.def_292 (not .def_291)))
(let ((.def_304 (or .def_289 .def_292)))
(let ((.def_306 (and .def_304 .def_305)))
(let ((.def_309 (and .def_306 .def_308)))
(let ((.def_303 (and .def_271 .def_277)))
(let ((.def_307 (and .def_303 .def_306)))
(let ((.def_310 (or .def_307 .def_309)))
(let ((.def_263 (* b.speed_y__AT1 b.delta__AT1)))
(let ((.def_264 (* 10.0 .def_263)))
(let ((.def_261 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_262 (* (- 49.0) .def_261)))
(let ((.def_265 (+ .def_262 .def_264)))
(let ((.def_266 (* 10.0 b.y__AT1)))
(let ((.def_268 (+ .def_266 .def_265)))
(let ((.def_283 (<= .def_268 0.0 )))
(let ((.def_297 (not .def_283)))
(let ((.def_281 (<= b.y__AT1 0.0 )))
(let ((.def_298 (or .def_281 .def_297)))
(let ((.def_251 (<= 0.0 b.y__AT1)))
(let ((.def_295 (not .def_251)))
(let ((.def_269 (<= 0.0 .def_268)))
(let ((.def_296 (or .def_269 .def_295)))
(let ((.def_299 (and .def_296 .def_298)))
(let ((.def_290 (not .def_289)))
(let ((.def_293 (or .def_290 .def_292)))
(let ((.def_294 (not .def_293)))
(let ((.def_300 (or .def_294 .def_299)))
(let ((.def_285 (not .def_269)))
(let ((.def_286 (or .def_251 .def_285)))
(let ((.def_282 (not .def_281)))
(let ((.def_284 (or .def_282 .def_283)))
(let ((.def_287 (and .def_284 .def_286)))
(let ((.def_272 (not .def_271)))
(let ((.def_279 (or .def_272 .def_278)))
(let ((.def_280 (not .def_279)))
(let ((.def_288 (or .def_280 .def_287)))
(let ((.def_301 (and .def_288 .def_300)))
(let ((.def_270 (and .def_251 .def_269)))
(let ((.def_302 (and .def_270 .def_301)))
(let ((.def_311 (and .def_302 .def_310)))
(let ((.def_61 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_58 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_62 (and .def_58 .def_61)))
(let ((.def_312 (and .def_62 .def_311)))
(let ((.def_32 (not b.counter.0__AT0)))
(let ((.def_30 (not b.counter.1__AT0)))
(let ((.def_33 (and .def_30 .def_32)))
(let ((.def_256 (or .def_33 b.counter.3__AT0)))
(let ((.def_38 (not b.counter.3__AT0)))
(let ((.def_35 (not b.counter.2__AT0)))
(let ((.def_36 (and .def_33 .def_35)))
(let ((.def_255 (or .def_36 .def_38)))
(let ((.def_257 (and .def_255 .def_256)))
(let ((.def_252 (and .def_62 .def_251)))
(let ((.def_248 (not b.EVENT.0__AT1)))
(let ((.def_246 (not b.EVENT.1__AT1)))
(let ((.def_249 (or .def_246 .def_248)))
(let ((.def_6 (not b.counter.0__AT1)))
(let ((.def_4 (not b.counter.1__AT1)))
(let ((.def_239 (or .def_4 .def_6)))
(let ((.def_243 (or b.counter.3__AT1 .def_239)))
(let ((.def_240 (or b.counter.2__AT1 .def_239)))
(let ((.def_9 (not b.counter.2__AT1)))
(let ((.def_238 (or .def_6 .def_9)))
(let ((.def_241 (and .def_238 .def_240)))
(let ((.def_12 (not b.counter.3__AT1)))
(let ((.def_242 (or .def_12 .def_241)))
(let ((.def_244 (and .def_242 .def_243)))
(let ((.def_250 (and .def_244 .def_249)))
(let ((.def_253 (and .def_250 .def_252)))
(let ((.def_233 (<= 0.0 b.delta__AT0)))
(let ((.def_51 (not b.EVENT.0__AT0)))
(let ((.def_49 (not b.EVENT.1__AT0)))
(let ((.def_179 (and .def_49 .def_51)))
(let ((.def_183 (not .def_179)))
(let ((.def_234 (or .def_183 .def_233)))
(let ((.def_235 (or b.EVENT.1__AT0 .def_234)))
(let ((.def_171 (= b.counter.0__AT1 b.counter.0__AT0)))
(let ((.def_169 (= b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_167 (= b.counter.2__AT1 b.counter.2__AT0)))
(let ((.def_166 (= b.counter.3__AT1 b.counter.3__AT0)))
(let ((.def_168 (and .def_166 .def_167)))
(let ((.def_170 (and .def_168 .def_169)))
(let ((.def_172 (and .def_170 .def_171)))
(let ((.def_230 (or .def_172 .def_183)))
(let ((.def_231 (or b.EVENT.1__AT0 .def_230)))
(let ((.def_219 (* (- 10.0) b.y__AT1)))
(let ((.def_72 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_73 (* 10.0 .def_72)))
(let ((.def_222 (+ .def_73 .def_219)))
(let ((.def_68 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_71 (* (- 49.0) .def_68)))
(let ((.def_223 (+ .def_71 .def_222)))
(let ((.def_75 (* 10.0 b.y__AT0)))
(let ((.def_224 (+ .def_75 .def_223)))
(let ((.def_225 (= .def_224 0.0 )))
(let ((.def_214 (* (- 5.0) b.speed_y__AT1)))
(let ((.def_82 (* (- 49.0) b.delta__AT0)))
(let ((.def_215 (+ .def_82 .def_214)))
(let ((.def_84 (* 5.0 b.speed_y__AT0)))
(let ((.def_216 (+ .def_84 .def_215)))
(let ((.def_217 (= .def_216 0.0 )))
(let ((.def_226 (and .def_217 .def_225)))
(let ((.def_227 (or .def_183 .def_226)))
(let ((.def_177 (= b.y__AT0 b.y__AT1)))
(let ((.def_165 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_210 (and .def_165 .def_177)))
(let ((.def_207 (= b.delta__AT0 0.0 )))
(let ((.def_208 (and .def_179 .def_207)))
(let ((.def_209 (not .def_208)))
(let ((.def_211 (or .def_209 .def_210)))
(let ((.def_212 (or b.EVENT.1__AT0 .def_211)))
(let ((.def_228 (and .def_212 .def_227)))
(let ((.def_189 (= b.time__AT0 b.time__AT1)))
(let ((.def_201 (and .def_177 .def_189)))
(let ((.def_202 (and .def_165 .def_201)))
(let ((.def_203 (and .def_172 .def_202)))
(let ((.def_204 (or .def_49 .def_203)))
(let ((.def_192 (* (- 1.0) b.time__AT1)))
(let ((.def_195 (+ b.delta__AT0 .def_192)))
(let ((.def_196 (+ b.time__AT0 .def_195)))
(let ((.def_197 (= .def_196 0.0 )))
(let ((.def_198 (or .def_183 .def_197)))
(let ((.def_199 (or b.EVENT.1__AT0 .def_198)))
(let ((.def_190 (or .def_179 .def_189)))
(let ((.def_191 (or b.EVENT.1__AT0 .def_190)))
(let ((.def_200 (and .def_191 .def_199)))
(let ((.def_205 (and .def_200 .def_204)))
(let ((.def_185 (= .def_183 b.event_is_timed__AT1)))
(let ((.def_182 (= b.event_is_timed__AT0 .def_179)))
(let ((.def_186 (and .def_182 .def_185)))
(let ((.def_173 (and .def_165 .def_172)))
(let ((.def_124 (= b.y__AT0 0.0 )))
(let ((.def_80 (<= 0.0 b.speed_y__AT0)))
(let ((.def_81 (not .def_80)))
(let ((.def_125 (and .def_81 .def_124)))
(let ((.def_174 (or .def_125 .def_173)))
(let ((.def_142 (or .def_6 b.counter.0__AT0)))
(let ((.def_141 (or b.counter.0__AT1 .def_32)))
(let ((.def_143 (and .def_141 .def_142)))
(let ((.def_144 (and .def_4 .def_143)))
(let ((.def_145 (or b.counter.1__AT0 .def_144)))
(let ((.def_140 (or b.counter.1__AT1 .def_30)))
(let ((.def_146 (and .def_140 .def_145)))
(let ((.def_157 (and .def_9 .def_146)))
(let ((.def_158 (or b.counter.2__AT0 .def_157)))
(let ((.def_152 (and .def_4 .def_32)))
(let ((.def_153 (or b.counter.1__AT0 .def_152)))
(let ((.def_154 (and .def_140 .def_153)))
(let ((.def_155 (and b.counter.2__AT1 .def_154)))
(let ((.def_156 (or .def_35 .def_155)))
(let ((.def_159 (and .def_156 .def_158)))
(let ((.def_160 (and b.counter.3__AT1 .def_159)))
(let ((.def_161 (or b.counter.3__AT0 .def_160)))
(let ((.def_147 (and b.counter.2__AT1 .def_146)))
(let ((.def_148 (or b.counter.2__AT0 .def_147)))
(let ((.def_136 (or b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_134 (and .def_4 b.counter.0__AT1)))
(let ((.def_135 (or .def_30 .def_134)))
(let ((.def_137 (and .def_135 .def_136)))
(let ((.def_138 (and .def_9 .def_137)))
(let ((.def_139 (or .def_35 .def_138)))
(let ((.def_149 (and .def_139 .def_148)))
(let ((.def_150 (and .def_12 .def_149)))
(let ((.def_151 (or .def_38 .def_150)))
(let ((.def_162 (and .def_151 .def_161)))
(let ((.def_130 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_128 (* (- 1.0) speed_loss__AT0)))
(let ((.def_129 (+ 1.0 .def_128)))
(let ((.def_131 (* .def_129 .def_130)))
(let ((.def_133 (= .def_131 b.speed_y__AT1)))
(let ((.def_163 (and .def_133 .def_162)))
(let ((.def_126 (not .def_125)))
(let ((.def_164 (or .def_126 .def_163)))
(let ((.def_175 (and .def_164 .def_174)))
(let ((.def_178 (and .def_175 .def_177)))
(let ((.def_180 (or .def_178 .def_179)))
(let ((.def_181 (or b.EVENT.1__AT0 .def_180)))
(let ((.def_187 (and .def_181 .def_186)))
(let ((.def_206 (and .def_187 .def_205)))
(let ((.def_229 (and .def_206 .def_228)))
(let ((.def_232 (and .def_229 .def_231)))
(let ((.def_236 (and .def_232 .def_235)))
(let ((.def_237 (and .def_49 .def_236)))
(let ((.def_254 (and .def_237 .def_253)))
(let ((.def_258 (and .def_254 .def_257)))
(let ((.def_101 (<= b.speed_y__AT0 0.0 )))
(let ((.def_86 (+ .def_84 .def_82)))
(let ((.def_99 (<= .def_86 0.0 )))
(let ((.def_118 (and .def_99 .def_101)))
(let ((.def_87 (<= 0.0 .def_86)))
(let ((.def_88 (not .def_87)))
(let ((.def_115 (or .def_80 .def_88)))
(let ((.def_102 (not .def_101)))
(let ((.def_114 (or .def_99 .def_102)))
(let ((.def_116 (and .def_114 .def_115)))
(let ((.def_119 (and .def_116 .def_118)))
(let ((.def_113 (and .def_80 .def_87)))
(let ((.def_117 (and .def_113 .def_116)))
(let ((.def_120 (or .def_117 .def_119)))
(let ((.def_74 (+ .def_71 .def_73)))
(let ((.def_77 (+ .def_75 .def_74)))
(let ((.def_93 (<= .def_77 0.0 )))
(let ((.def_107 (not .def_93)))
(let ((.def_91 (<= b.y__AT0 0.0 )))
(let ((.def_108 (or .def_91 .def_107)))
(let ((.def_54 (<= 0.0 b.y__AT0)))
(let ((.def_105 (not .def_54)))
(let ((.def_78 (<= 0.0 .def_77)))
(let ((.def_106 (or .def_78 .def_105)))
(let ((.def_109 (and .def_106 .def_108)))
(let ((.def_100 (not .def_99)))
(let ((.def_103 (or .def_100 .def_102)))
(let ((.def_104 (not .def_103)))
(let ((.def_110 (or .def_104 .def_109)))
(let ((.def_95 (not .def_78)))
(let ((.def_96 (or .def_54 .def_95)))
(let ((.def_92 (not .def_91)))
(let ((.def_94 (or .def_92 .def_93)))
(let ((.def_97 (and .def_94 .def_96)))
(let ((.def_89 (or .def_81 .def_88)))
(let ((.def_90 (not .def_89)))
(let ((.def_98 (or .def_90 .def_97)))
(let ((.def_111 (and .def_98 .def_110)))
(let ((.def_79 (and .def_54 .def_78)))
(let ((.def_112 (and .def_79 .def_111)))
(let ((.def_121 (and .def_112 .def_120)))
(let ((.def_122 (and .def_62 .def_121)))
(let ((.def_63 (and .def_54 .def_62)))
(let ((.def_52 (or .def_49 .def_51)))
(let ((.def_42 (or .def_30 .def_32)))
(let ((.def_46 (or b.counter.3__AT0 .def_42)))
(let ((.def_43 (or b.counter.2__AT0 .def_42)))
(let ((.def_41 (or .def_32 .def_35)))
(let ((.def_44 (and .def_41 .def_43)))
(let ((.def_45 (or .def_38 .def_44)))
(let ((.def_47 (and .def_45 .def_46)))
(let ((.def_53 (and .def_47 .def_52)))
(let ((.def_64 (and .def_53 .def_63)))
(let ((.def_39 (and .def_36 .def_38)))
(let ((.def_27 (= b.speed_y__AT0 0.0 )))
(let ((.def_24 (= b.y__AT0 10.0 )))
(let ((.def_19 (= b.time__AT0 0.0 )))
(let ((.def_21 (and .def_19 b.event_is_timed__AT0)))
(let ((.def_25 (and .def_21 .def_24)))
(let ((.def_28 (and .def_25 .def_27)))
(let ((.def_40 (and .def_28 .def_39)))
(let ((.def_65 (and .def_40 .def_64)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_14 (or .def_7 b.counter.3__AT1)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_13 (or .def_10 .def_12)))
(let ((.def_15 (and .def_13 .def_14)))
(let ((.def_16 (not .def_15)))
(let ((.def_66 (and .def_16 .def_65)))
(let ((.def_123 (and .def_66 .def_122)))
(let ((.def_259 (and .def_123 .def_258)))
(let ((.def_313 (and .def_259 .def_312)))
.def_313)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)