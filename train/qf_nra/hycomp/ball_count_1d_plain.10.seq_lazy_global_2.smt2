(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 2 and uses the quantifier free encoding encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:50:39 2012
(declare-fun b.time__AT2 () Real)
(declare-fun b.delta__AT1 () Real)
(declare-fun b.delta__AT2 () Real)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.EVENT.0__AT2 () Bool)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.y__AT2 () Real)
(declare-fun b.EVENT.1__AT2 () Bool)
(declare-fun b.time__AT1 () Real)
(declare-fun b.speed_y__AT1 () Real)
(declare-fun b.time__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.speed_y__AT2 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.counter.2__AT2 () Bool)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.counter.3__AT2 () Bool)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.0__AT2 () Bool)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.1__AT2 () Bool)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.event_is_timed__AT2 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.y__AT1 () Real)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(assert (let ((.def_408 (* (- 49.0) b.delta__AT2)))
(let ((.def_409 (* 5.0 b.speed_y__AT2)))
(let ((.def_411 (+ .def_409 .def_408)))
(let ((.def_415 (<= .def_411 0.0 )))
(let ((.def_414 (<= b.speed_y__AT2 0.0 )))
(let ((.def_416 (and .def_414 .def_415)))
(let ((.def_412 (<= 0.0 .def_411)))
(let ((.def_407 (<= 0.0 b.speed_y__AT2)))
(let ((.def_413 (and .def_407 .def_412)))
(let ((.def_417 (or .def_413 .def_416)))
(let ((.def_399 (* b.speed_y__AT2 b.delta__AT2)))
(let ((.def_400 (* 10.0 .def_399)))
(let ((.def_397 (* b.delta__AT2 b.delta__AT2)))
(let ((.def_398 (* (- 49.0) .def_397)))
(let ((.def_401 (+ .def_398 .def_400)))
(let ((.def_402 (* 10.0 b.y__AT2)))
(let ((.def_404 (+ .def_402 .def_401)))
(let ((.def_405 (<= 0.0 .def_404)))
(let ((.def_387 (<= 0.0 b.y__AT2)))
(let ((.def_406 (and .def_387 .def_405)))
(let ((.def_418 (and .def_406 .def_417)))
(let ((.def_57 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_54 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_58 (and .def_54 .def_57)))
(let ((.def_419 (and .def_58 .def_418)))
(let ((.def_115 (not b.counter.0__AT1)))
(let ((.def_391 (or b.counter.1__AT1 .def_115)))
(let ((.def_110 (not b.counter.2__AT1)))
(let ((.def_392 (or .def_110 .def_391)))
(let ((.def_393 (or b.counter.3__AT1 .def_392)))
(let ((.def_388 (and .def_58 .def_387)))
(let ((.def_384 (not b.EVENT.0__AT2)))
(let ((.def_382 (not b.EVENT.1__AT2)))
(let ((.def_385 (or .def_382 .def_384)))
(let ((.def_269 (not b.counter.1__AT2)))
(let ((.def_4 (not b.counter.0__AT2)))
(let ((.def_375 (or .def_4 .def_269)))
(let ((.def_379 (or b.counter.3__AT2 .def_375)))
(let ((.def_376 (or b.counter.2__AT2 .def_375)))
(let ((.def_8 (not b.counter.2__AT2)))
(let ((.def_374 (or .def_4 .def_8)))
(let ((.def_377 (and .def_374 .def_376)))
(let ((.def_286 (not b.counter.3__AT2)))
(let ((.def_378 (or .def_286 .def_377)))
(let ((.def_380 (and .def_378 .def_379)))
(let ((.def_386 (and .def_380 .def_385)))
(let ((.def_389 (and .def_386 .def_388)))
(let ((.def_369 (<= 0.0 b.delta__AT1)))
(let ((.def_224 (not b.EVENT.0__AT1)))
(let ((.def_222 (not b.EVENT.1__AT1)))
(let ((.def_316 (and .def_222 .def_224)))
(let ((.def_320 (not .def_316)))
(let ((.def_370 (or .def_320 .def_369)))
(let ((.def_371 (or b.EVENT.1__AT1 .def_370)))
(let ((.def_308 (= b.counter.0__AT2 b.counter.0__AT1)))
(let ((.def_306 (= b.counter.1__AT2 b.counter.1__AT1)))
(let ((.def_304 (= b.counter.2__AT2 b.counter.2__AT1)))
(let ((.def_303 (= b.counter.3__AT2 b.counter.3__AT1)))
(let ((.def_305 (and .def_303 .def_304)))
(let ((.def_307 (and .def_305 .def_306)))
(let ((.def_309 (and .def_307 .def_308)))
(let ((.def_366 (or .def_309 .def_320)))
(let ((.def_367 (or b.EVENT.1__AT1 .def_366)))
(let ((.def_354 (* (- 10.0) b.y__AT2)))
(let ((.def_239 (* b.speed_y__AT1 b.delta__AT1)))
(let ((.def_240 (* 10.0 .def_239)))
(let ((.def_358 (+ .def_240 .def_354)))
(let ((.def_237 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_238 (* (- 49.0) .def_237)))
(let ((.def_359 (+ .def_238 .def_358)))
(let ((.def_242 (* 10.0 b.y__AT1)))
(let ((.def_360 (+ .def_242 .def_359)))
(let ((.def_361 (= .def_360 0.0 )))
(let ((.def_350 (* (- 5.0) b.speed_y__AT2)))
(let ((.def_248 (* (- 49.0) b.delta__AT1)))
(let ((.def_351 (+ .def_248 .def_350)))
(let ((.def_249 (* 5.0 b.speed_y__AT1)))
(let ((.def_352 (+ .def_249 .def_351)))
(let ((.def_353 (= .def_352 0.0 )))
(let ((.def_362 (and .def_353 .def_361)))
(let ((.def_363 (or .def_320 .def_362)))
(let ((.def_314 (= b.y__AT1 b.y__AT2)))
(let ((.def_302 (= b.speed_y__AT1 b.speed_y__AT2)))
(let ((.def_347 (and .def_302 .def_314)))
(let ((.def_344 (= b.delta__AT1 0.0 )))
(let ((.def_345 (and .def_316 .def_344)))
(let ((.def_346 (not .def_345)))
(let ((.def_348 (or .def_346 .def_347)))
(let ((.def_349 (or b.EVENT.1__AT1 .def_348)))
(let ((.def_364 (and .def_349 .def_363)))
(let ((.def_326 (= b.time__AT1 b.time__AT2)))
(let ((.def_338 (and .def_314 .def_326)))
(let ((.def_339 (and .def_302 .def_338)))
(let ((.def_340 (and .def_309 .def_339)))
(let ((.def_341 (or .def_222 .def_340)))
(let ((.def_329 (* (- 1.0) b.time__AT2)))
(let ((.def_332 (+ b.delta__AT1 .def_329)))
(let ((.def_333 (+ b.time__AT1 .def_332)))
(let ((.def_334 (= .def_333 0.0 )))
(let ((.def_335 (or .def_320 .def_334)))
(let ((.def_336 (or b.EVENT.1__AT1 .def_335)))
(let ((.def_327 (or .def_316 .def_326)))
(let ((.def_328 (or b.EVENT.1__AT1 .def_327)))
(let ((.def_337 (and .def_328 .def_336)))
(let ((.def_342 (and .def_337 .def_341)))
(let ((.def_322 (= .def_320 b.event_is_timed__AT2)))
(let ((.def_319 (= b.event_is_timed__AT1 .def_316)))
(let ((.def_323 (and .def_319 .def_322)))
(let ((.def_310 (and .def_302 .def_309)))
(let ((.def_247 (<= 0.0 b.speed_y__AT1)))
(let ((.def_262 (not .def_247)))
(let ((.def_261 (= b.y__AT1 0.0 )))
(let ((.def_263 (and .def_261 .def_262)))
(let ((.def_311 (or .def_263 .def_310)))
(let ((.def_278 (or .def_4 b.counter.0__AT1)))
(let ((.def_277 (or b.counter.0__AT2 .def_115)))
(let ((.def_279 (and .def_277 .def_278)))
(let ((.def_280 (and .def_269 .def_279)))
(let ((.def_281 (or b.counter.1__AT1 .def_280)))
(let ((.def_103 (not b.counter.1__AT1)))
(let ((.def_276 (or b.counter.1__AT2 .def_103)))
(let ((.def_282 (and .def_276 .def_281)))
(let ((.def_294 (and .def_8 .def_282)))
(let ((.def_295 (or b.counter.2__AT1 .def_294)))
(let ((.def_289 (and .def_115 .def_269)))
(let ((.def_290 (or b.counter.1__AT1 .def_289)))
(let ((.def_291 (and .def_276 .def_290)))
(let ((.def_292 (and b.counter.2__AT2 .def_291)))
(let ((.def_293 (or .def_110 .def_292)))
(let ((.def_296 (and .def_293 .def_295)))
(let ((.def_297 (and b.counter.3__AT2 .def_296)))
(let ((.def_298 (or b.counter.3__AT1 .def_297)))
(let ((.def_283 (and b.counter.2__AT2 .def_282)))
(let ((.def_284 (or b.counter.2__AT1 .def_283)))
(let ((.def_272 (or b.counter.1__AT2 b.counter.1__AT1)))
(let ((.def_270 (and b.counter.0__AT2 .def_269)))
(let ((.def_271 (or .def_103 .def_270)))
(let ((.def_273 (and .def_271 .def_272)))
(let ((.def_274 (and .def_8 .def_273)))
(let ((.def_275 (or .def_110 .def_274)))
(let ((.def_285 (and .def_275 .def_284)))
(let ((.def_287 (and .def_285 .def_286)))
(let ((.def_125 (not b.counter.3__AT1)))
(let ((.def_288 (or .def_125 .def_287)))
(let ((.def_299 (and .def_288 .def_298)))
(let ((.def_265 (* (- 1.0) b.speed_y__AT1)))
(let ((.def_96 (* (- 1.0) speed_loss__AT0)))
(let ((.def_97 (+ 1.0 .def_96)))
(let ((.def_266 (* .def_97 .def_265)))
(let ((.def_268 (= .def_266 b.speed_y__AT2)))
(let ((.def_300 (and .def_268 .def_299)))
(let ((.def_264 (not .def_263)))
(let ((.def_301 (or .def_264 .def_300)))
(let ((.def_312 (and .def_301 .def_311)))
(let ((.def_315 (and .def_312 .def_314)))
(let ((.def_317 (or .def_315 .def_316)))
(let ((.def_318 (or b.EVENT.1__AT1 .def_317)))
(let ((.def_324 (and .def_318 .def_323)))
(let ((.def_343 (and .def_324 .def_342)))
(let ((.def_365 (and .def_343 .def_364)))
(let ((.def_368 (and .def_365 .def_367)))
(let ((.def_372 (and .def_368 .def_371)))
(let ((.def_373 (and .def_222 .def_372)))
(let ((.def_390 (and .def_373 .def_389)))
(let ((.def_394 (and .def_390 .def_393)))
(let ((.def_251 (+ .def_249 .def_248)))
(let ((.def_255 (<= .def_251 0.0 )))
(let ((.def_254 (<= b.speed_y__AT1 0.0 )))
(let ((.def_256 (and .def_254 .def_255)))
(let ((.def_252 (<= 0.0 .def_251)))
(let ((.def_253 (and .def_247 .def_252)))
(let ((.def_257 (or .def_253 .def_256)))
(let ((.def_241 (+ .def_238 .def_240)))
(let ((.def_244 (+ .def_242 .def_241)))
(let ((.def_245 (<= 0.0 .def_244)))
(let ((.def_227 (<= 0.0 b.y__AT1)))
(let ((.def_246 (and .def_227 .def_245)))
(let ((.def_258 (and .def_246 .def_257)))
(let ((.def_259 (and .def_58 .def_258)))
(let ((.def_28 (not b.counter.0__AT0)))
(let ((.def_231 (or b.counter.1__AT0 .def_28)))
(let ((.def_31 (not b.counter.2__AT0)))
(let ((.def_232 (or .def_31 .def_231)))
(let ((.def_233 (or b.counter.3__AT0 .def_232)))
(let ((.def_228 (and .def_58 .def_227)))
(let ((.def_225 (or .def_222 .def_224)))
(let ((.def_215 (or .def_103 .def_115)))
(let ((.def_219 (or b.counter.3__AT1 .def_215)))
(let ((.def_216 (or b.counter.2__AT1 .def_215)))
(let ((.def_214 (or .def_110 .def_115)))
(let ((.def_217 (and .def_214 .def_216)))
(let ((.def_218 (or .def_125 .def_217)))
(let ((.def_220 (and .def_218 .def_219)))
(let ((.def_226 (and .def_220 .def_225)))
(let ((.def_229 (and .def_226 .def_228)))
(let ((.def_209 (<= 0.0 b.delta__AT0)))
(let ((.def_47 (not b.EVENT.0__AT0)))
(let ((.def_45 (not b.EVENT.1__AT0)))
(let ((.def_155 (and .def_45 .def_47)))
(let ((.def_159 (not .def_155)))
(let ((.def_210 (or .def_159 .def_209)))
(let ((.def_211 (or b.EVENT.1__AT0 .def_210)))
(let ((.def_147 (= b.counter.0__AT0 b.counter.0__AT1)))
(let ((.def_145 (= b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_143 (= b.counter.2__AT0 b.counter.2__AT1)))
(let ((.def_142 (= b.counter.3__AT0 b.counter.3__AT1)))
(let ((.def_144 (and .def_142 .def_143)))
(let ((.def_146 (and .def_144 .def_145)))
(let ((.def_148 (and .def_146 .def_147)))
(let ((.def_206 (or .def_148 .def_159)))
(let ((.def_207 (or b.EVENT.1__AT0 .def_206)))
(let ((.def_195 (* (- 10.0) b.y__AT1)))
(let ((.def_68 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_69 (* 10.0 .def_68)))
(let ((.def_198 (+ .def_69 .def_195)))
(let ((.def_64 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_67 (* (- 49.0) .def_64)))
(let ((.def_199 (+ .def_67 .def_198)))
(let ((.def_71 (* 10.0 b.y__AT0)))
(let ((.def_200 (+ .def_71 .def_199)))
(let ((.def_201 (= .def_200 0.0 )))
(let ((.def_190 (* (- 5.0) b.speed_y__AT1)))
(let ((.def_77 (* (- 49.0) b.delta__AT0)))
(let ((.def_191 (+ .def_77 .def_190)))
(let ((.def_79 (* 5.0 b.speed_y__AT0)))
(let ((.def_192 (+ .def_79 .def_191)))
(let ((.def_193 (= .def_192 0.0 )))
(let ((.def_202 (and .def_193 .def_201)))
(let ((.def_203 (or .def_159 .def_202)))
(let ((.def_153 (= b.y__AT0 b.y__AT1)))
(let ((.def_141 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_186 (and .def_141 .def_153)))
(let ((.def_183 (= b.delta__AT0 0.0 )))
(let ((.def_184 (and .def_155 .def_183)))
(let ((.def_185 (not .def_184)))
(let ((.def_187 (or .def_185 .def_186)))
(let ((.def_188 (or b.EVENT.1__AT0 .def_187)))
(let ((.def_204 (and .def_188 .def_203)))
(let ((.def_165 (= b.time__AT0 b.time__AT1)))
(let ((.def_177 (and .def_153 .def_165)))
(let ((.def_178 (and .def_141 .def_177)))
(let ((.def_179 (and .def_148 .def_178)))
(let ((.def_180 (or .def_45 .def_179)))
(let ((.def_168 (* (- 1.0) b.time__AT1)))
(let ((.def_171 (+ b.delta__AT0 .def_168)))
(let ((.def_172 (+ b.time__AT0 .def_171)))
(let ((.def_173 (= .def_172 0.0 )))
(let ((.def_174 (or .def_159 .def_173)))
(let ((.def_175 (or b.EVENT.1__AT0 .def_174)))
(let ((.def_166 (or .def_155 .def_165)))
(let ((.def_167 (or b.EVENT.1__AT0 .def_166)))
(let ((.def_176 (and .def_167 .def_175)))
(let ((.def_181 (and .def_176 .def_180)))
(let ((.def_161 (= .def_159 b.event_is_timed__AT1)))
(let ((.def_158 (= b.event_is_timed__AT0 .def_155)))
(let ((.def_162 (and .def_158 .def_161)))
(let ((.def_149 (and .def_141 .def_148)))
(let ((.def_76 (<= 0.0 b.speed_y__AT0)))
(let ((.def_92 (not .def_76)))
(let ((.def_91 (= b.y__AT0 0.0 )))
(let ((.def_93 (and .def_91 .def_92)))
(let ((.def_150 (or .def_93 .def_149)))
(let ((.def_116 (or b.counter.0__AT0 .def_115)))
(let ((.def_114 (or .def_28 b.counter.0__AT1)))
(let ((.def_117 (and .def_114 .def_116)))
(let ((.def_118 (and .def_103 .def_117)))
(let ((.def_119 (or b.counter.1__AT0 .def_118)))
(let ((.def_26 (not b.counter.1__AT0)))
(let ((.def_113 (or .def_26 b.counter.1__AT1)))
(let ((.def_120 (and .def_113 .def_119)))
(let ((.def_133 (and .def_110 .def_120)))
(let ((.def_134 (or b.counter.2__AT0 .def_133)))
(let ((.def_128 (and .def_28 .def_103)))
(let ((.def_129 (or b.counter.1__AT0 .def_128)))
(let ((.def_130 (and .def_113 .def_129)))
(let ((.def_131 (and b.counter.2__AT1 .def_130)))
(let ((.def_132 (or .def_31 .def_131)))
(let ((.def_135 (and .def_132 .def_134)))
(let ((.def_136 (and b.counter.3__AT1 .def_135)))
(let ((.def_137 (or b.counter.3__AT0 .def_136)))
(let ((.def_121 (and b.counter.2__AT1 .def_120)))
(let ((.def_122 (or b.counter.2__AT0 .def_121)))
(let ((.def_107 (or b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_105 (and .def_103 b.counter.0__AT1)))
(let ((.def_106 (or .def_26 .def_105)))
(let ((.def_108 (and .def_106 .def_107)))
(let ((.def_111 (and .def_108 .def_110)))
(let ((.def_112 (or .def_31 .def_111)))
(let ((.def_123 (and .def_112 .def_122)))
(let ((.def_126 (and .def_123 .def_125)))
(let ((.def_34 (not b.counter.3__AT0)))
(let ((.def_127 (or .def_34 .def_126)))
(let ((.def_138 (and .def_127 .def_137)))
(let ((.def_98 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_99 (* .def_97 .def_98)))
(let ((.def_101 (= .def_99 b.speed_y__AT1)))
(let ((.def_139 (and .def_101 .def_138)))
(let ((.def_94 (not .def_93)))
(let ((.def_140 (or .def_94 .def_139)))
(let ((.def_151 (and .def_140 .def_150)))
(let ((.def_154 (and .def_151 .def_153)))
(let ((.def_156 (or .def_154 .def_155)))
(let ((.def_157 (or b.EVENT.1__AT0 .def_156)))
(let ((.def_163 (and .def_157 .def_162)))
(let ((.def_182 (and .def_163 .def_181)))
(let ((.def_205 (and .def_182 .def_204)))
(let ((.def_208 (and .def_205 .def_207)))
(let ((.def_212 (and .def_208 .def_211)))
(let ((.def_213 (and .def_45 .def_212)))
(let ((.def_230 (and .def_213 .def_229)))
(let ((.def_234 (and .def_230 .def_233)))
(let ((.def_81 (+ .def_79 .def_77)))
(let ((.def_85 (<= .def_81 0.0 )))
(let ((.def_84 (<= b.speed_y__AT0 0.0 )))
(let ((.def_86 (and .def_84 .def_85)))
(let ((.def_82 (<= 0.0 .def_81)))
(let ((.def_83 (and .def_76 .def_82)))
(let ((.def_87 (or .def_83 .def_86)))
(let ((.def_70 (+ .def_67 .def_69)))
(let ((.def_73 (+ .def_71 .def_70)))
(let ((.def_74 (<= 0.0 .def_73)))
(let ((.def_50 (<= 0.0 b.y__AT0)))
(let ((.def_75 (and .def_50 .def_74)))
(let ((.def_88 (and .def_75 .def_87)))
(let ((.def_89 (and .def_58 .def_88)))
(let ((.def_59 (and .def_50 .def_58)))
(let ((.def_48 (or .def_45 .def_47)))
(let ((.def_38 (or .def_26 .def_28)))
(let ((.def_42 (or b.counter.3__AT0 .def_38)))
(let ((.def_39 (or b.counter.2__AT0 .def_38)))
(let ((.def_37 (or .def_28 .def_31)))
(let ((.def_40 (and .def_37 .def_39)))
(let ((.def_41 (or .def_34 .def_40)))
(let ((.def_43 (and .def_41 .def_42)))
(let ((.def_49 (and .def_43 .def_48)))
(let ((.def_60 (and .def_49 .def_59)))
(let ((.def_29 (and .def_26 .def_28)))
(let ((.def_32 (and .def_29 .def_31)))
(let ((.def_35 (and .def_32 .def_34)))
(let ((.def_23 (= b.speed_y__AT0 0.0 )))
(let ((.def_20 (= b.y__AT0 10.0 )))
(let ((.def_15 (= b.time__AT0 0.0 )))
(let ((.def_17 (and .def_15 b.event_is_timed__AT0)))
(let ((.def_21 (and .def_17 .def_20)))
(let ((.def_24 (and .def_21 .def_23)))
(let ((.def_36 (and .def_24 .def_35)))
(let ((.def_61 (and .def_36 .def_60)))
(let ((.def_6 (or .def_4 b.counter.1__AT2)))
(let ((.def_9 (or .def_6 .def_8)))
(let ((.def_11 (or .def_9 b.counter.3__AT2)))
(let ((.def_12 (not .def_11)))
(let ((.def_62 (and .def_12 .def_61)))
(let ((.def_90 (and .def_62 .def_89)))
(let ((.def_235 (and .def_90 .def_234)))
(let ((.def_260 (and .def_235 .def_259)))
(let ((.def_395 (and .def_260 .def_394)))
(let ((.def_420 (and .def_395 .def_419)))
.def_420))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
