(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 1 and uses the quantifier free encoding with equivalences encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:37:14 2012
(declare-fun b.x__AT0 () Real)
(declare-fun b.speed_y__AT1 () Real)
(declare-fun b.delta__AT1 () Real)
(declare-fun b.bool_atom__AT0 () Bool)
(declare-fun b.y__AT0 () Real)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.y__AT1 () Real)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.speed_x__AT1 () Real)
(declare-fun b.time__AT1 () Real)
(declare-fun b.x__AT1 () Real)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.speed_loss__AT0 () Real)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.speed_x__AT0 () Real)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.bool_atom__AT1 () Bool)
(assert (let ((.def_482 (* (- 49.0) b.delta__AT1)))
(let ((.def_483 (* 5.0 b.speed_y__AT1)))
(let ((.def_485 (+ .def_483 .def_482)))
(let ((.def_488 (<= 0.0 .def_485)))
(let ((.def_416 (* b.y__AT1 b.speed_y__AT1)))
(let ((.def_456 (* 2.0 .def_416)))
(let ((.def_457 (+ b.speed_x__AT1 .def_456)))
(let ((.def_461 (<= .def_457 0.0 )))
(let ((.def_489 (and .def_461 .def_488)))
(let ((.def_486 (<= .def_485 0.0 )))
(let ((.def_458 (<= 0.0 .def_457)))
(let ((.def_487 (and .def_458 .def_486)))
(let ((.def_490 (or .def_487 .def_489)))
(let ((.def_405 (* (- (/ 49 10)) b.speed_y__AT1)))
(let ((.def_434 (* 3.0 .def_405)))
(let ((.def_466 (* 2.0 .def_434)))
(let ((.def_467 (* b.delta__AT1 .def_466)))
(let ((.def_468 (* (- 50.0) .def_467)))
(let ((.def_400 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_465 (* (- 7203.0) .def_400)))
(let ((.def_469 (+ .def_465 .def_468)))
(let ((.def_274 (* b.speed_y__AT1 b.speed_y__AT1)))
(let ((.def_470 (* (- 50.0) .def_274)))
(let ((.def_474 (+ .def_470 .def_469)))
(let ((.def_472 (* 490.0 b.y__AT1)))
(let ((.def_475 (+ .def_472 .def_474)))
(let ((.def_478 (<= 0.0 .def_475)))
(let ((.def_479 (and .def_461 .def_478)))
(let ((.def_476 (<= .def_475 0.0 )))
(let ((.def_477 (and .def_458 .def_476)))
(let ((.def_480 (or .def_477 .def_479)))
(let ((.def_412 (* (- (/ 49 10)) b.y__AT1)))
(let ((.def_442 (* 2.0 .def_412)))
(let ((.def_443 (* b.delta__AT1 .def_442)))
(let ((.def_444 (* 50.0 .def_443)))
(let ((.def_438 (* 2.0 .def_274)))
(let ((.def_439 (* b.delta__AT1 .def_438)))
(let ((.def_440 (* 25.0 .def_439)))
(let ((.def_450 (+ .def_440 .def_444)))
(let ((.def_435 (* .def_400 .def_434)))
(let ((.def_436 (* 50.0 .def_435)))
(let ((.def_451 (+ .def_436 .def_450)))
(let ((.def_446 (* 50.0 .def_416)))
(let ((.def_452 (+ .def_446 .def_451)))
(let ((.def_401 (* b.delta__AT1 .def_400)))
(let ((.def_433 (* 2401.0 .def_401)))
(let ((.def_453 (+ .def_433 .def_452)))
(let ((.def_448 (* 25.0 b.speed_x__AT1)))
(let ((.def_454 (+ .def_448 .def_453)))
(let ((.def_460 (<= .def_454 0.0 )))
(let ((.def_462 (and .def_460 .def_461)))
(let ((.def_455 (<= 0.0 .def_454)))
(let ((.def_459 (and .def_455 .def_458)))
(let ((.def_463 (or .def_459 .def_462)))
(let ((.def_417 (* b.delta__AT1 .def_416)))
(let ((.def_418 (* 200.0 .def_417)))
(let ((.def_413 (* .def_400 .def_412)))
(let ((.def_414 (* 200.0 .def_413)))
(let ((.def_424 (+ .def_414 .def_418)))
(let ((.def_409 (* .def_274 .def_400)))
(let ((.def_410 (* 100.0 .def_409)))
(let ((.def_425 (+ .def_410 .def_424)))
(let ((.def_406 (* .def_401 .def_405)))
(let ((.def_407 (* 200.0 .def_406)))
(let ((.def_426 (+ .def_407 .def_425)))
(let ((.def_402 (* b.delta__AT1 .def_401)))
(let ((.def_403 (* 2401.0 .def_402)))
(let ((.def_427 (+ .def_403 .def_426)))
(let ((.def_398 (* b.speed_x__AT1 b.delta__AT1)))
(let ((.def_399 (* 100.0 .def_398)))
(let ((.def_428 (+ .def_399 .def_427)))
(let ((.def_379 (* b.y__AT1 b.y__AT1)))
(let ((.def_420 (* 100.0 .def_379)))
(let ((.def_429 (+ .def_420 .def_428)))
(let ((.def_422 (* 100.0 b.x__AT1)))
(let ((.def_430 (+ .def_422 .def_429)))
(let ((.def_431 (<= 0.0 .def_430)))
(let ((.def_380 (+ b.x__AT1 .def_379)))
(let ((.def_381 (<= 0.0 .def_380)))
(let ((.def_432 (and .def_381 .def_431)))
(let ((.def_464 (and .def_432 .def_463)))
(let ((.def_481 (and .def_464 .def_480)))
(let ((.def_491 (and .def_481 .def_490)))
(let ((.def_77 (<= b.speed_loss__AT0 (/ 1 2))))
(let ((.def_74 (<= (/ 1 10) b.speed_loss__AT0)))
(let ((.def_78 (and .def_74 .def_77)))
(let ((.def_492 (and .def_78 .def_491)))
(let ((.def_47 (not b.counter.0__AT0)))
(let ((.def_390 (or b.counter.1__AT0 .def_47)))
(let ((.def_391 (or b.counter.2__AT0 .def_390)))
(let ((.def_50 (not b.counter.2__AT0)))
(let ((.def_45 (not b.counter.1__AT0)))
(let ((.def_48 (and .def_45 .def_47)))
(let ((.def_389 (or .def_48 .def_50)))
(let ((.def_392 (and .def_389 .def_391)))
(let ((.def_393 (or b.counter.3__AT0 .def_392)))
(let ((.def_386 (or .def_48 b.counter.2__AT0)))
(let ((.def_385 (or .def_45 .def_50)))
(let ((.def_387 (and .def_385 .def_386)))
(let ((.def_53 (not b.counter.3__AT0)))
(let ((.def_388 (or .def_53 .def_387)))
(let ((.def_394 (and .def_388 .def_393)))
(let ((.def_382 (and .def_78 .def_381)))
(let ((.def_376 (not b.EVENT.0__AT1)))
(let ((.def_374 (not b.EVENT.1__AT1)))
(let ((.def_377 (or .def_374 .def_376)))
(let ((.def_9 (not b.counter.0__AT1)))
(let ((.def_6 (not b.counter.1__AT1)))
(let ((.def_367 (or .def_6 .def_9)))
(let ((.def_371 (or b.counter.3__AT1 .def_367)))
(let ((.def_368 (or b.counter.2__AT1 .def_367)))
(let ((.def_4 (not b.counter.2__AT1)))
(let ((.def_366 (or .def_4 .def_9)))
(let ((.def_369 (and .def_366 .def_368)))
(let ((.def_14 (not b.counter.3__AT1)))
(let ((.def_370 (or .def_14 .def_369)))
(let ((.def_372 (and .def_370 .def_371)))
(let ((.def_378 (and .def_372 .def_377)))
(let ((.def_383 (and .def_378 .def_382)))
(let ((.def_361 (<= 0.0 b.delta__AT0)))
(let ((.def_66 (not b.EVENT.0__AT0)))
(let ((.def_64 (not b.EVENT.1__AT0)))
(let ((.def_197 (and .def_64 .def_66)))
(let ((.def_199 (not .def_197)))
(let ((.def_362 (or .def_199 .def_361)))
(let ((.def_363 (or b.EVENT.1__AT0 .def_362)))
(let ((.def_239 (= b.bool_atom__AT0 b.bool_atom__AT1)))
(let ((.def_234 (= b.counter.0__AT1 b.counter.0__AT0)))
(let ((.def_232 (= b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_230 (= b.counter.2__AT1 b.counter.2__AT0)))
(let ((.def_229 (= b.counter.3__AT1 b.counter.3__AT0)))
(let ((.def_231 (and .def_229 .def_230)))
(let ((.def_233 (and .def_231 .def_232)))
(let ((.def_235 (and .def_233 .def_234)))
(let ((.def_357 (and .def_235 .def_239)))
(let ((.def_358 (or .def_199 .def_357)))
(let ((.def_359 (or b.EVENT.1__AT0 .def_358)))
(let ((.def_339 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_340 (* 10.0 .def_339)))
(let ((.def_344 (* (- 10.0) b.y__AT1)))
(let ((.def_348 (+ .def_344 .def_340)))
(let ((.def_87 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_341 (* (- 49.0) .def_87)))
(let ((.def_349 (+ .def_341 .def_348)))
(let ((.def_346 (* 10.0 b.y__AT0)))
(let ((.def_350 (+ .def_346 .def_349)))
(let ((.def_351 (= .def_350 0.0 )))
(let ((.def_334 (* (- 5.0) b.speed_y__AT1)))
(let ((.def_184 (* (- 49.0) b.delta__AT0)))
(let ((.def_335 (+ .def_184 .def_334)))
(let ((.def_186 (* 5.0 b.speed_y__AT0)))
(let ((.def_336 (+ .def_186 .def_335)))
(let ((.def_337 (= .def_336 0.0 )))
(let ((.def_329 (* (- 1.0) b.x__AT1)))
(let ((.def_84 (* b.speed_x__AT0 b.delta__AT0)))
(let ((.def_330 (+ .def_84 .def_329)))
(let ((.def_331 (+ b.x__AT0 .def_330)))
(let ((.def_332 (= .def_331 0.0 )))
(let ((.def_338 (and .def_332 .def_337)))
(let ((.def_352 (and .def_338 .def_351)))
(let ((.def_224 (= b.speed_x__AT0 b.speed_x__AT1)))
(let ((.def_353 (and .def_224 .def_352)))
(let ((.def_354 (or .def_199 .def_353)))
(let ((.def_221 (= b.y__AT0 b.y__AT1)))
(let ((.def_218 (= b.x__AT0 b.x__AT1)))
(let ((.def_323 (and .def_218 .def_221)))
(let ((.def_324 (and .def_224 .def_323)))
(let ((.def_227 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_325 (and .def_227 .def_324)))
(let ((.def_320 (= b.delta__AT0 0.0 )))
(let ((.def_321 (and .def_197 .def_320)))
(let ((.def_322 (not .def_321)))
(let ((.def_326 (or .def_322 .def_325)))
(let ((.def_327 (or b.EVENT.1__AT0 .def_326)))
(let ((.def_312 (and .def_224 .def_227)))
(let ((.def_313 (and .def_235 .def_312)))
(let ((.def_314 (or b.bool_atom__AT0 .def_313)))
(let ((.def_288 (or .def_9 b.counter.0__AT0)))
(let ((.def_287 (or b.counter.0__AT1 .def_47)))
(let ((.def_289 (and .def_287 .def_288)))
(let ((.def_290 (and .def_6 .def_289)))
(let ((.def_291 (or b.counter.1__AT0 .def_290)))
(let ((.def_286 (or b.counter.1__AT1 .def_45)))
(let ((.def_292 (and .def_286 .def_291)))
(let ((.def_303 (and .def_4 .def_292)))
(let ((.def_304 (or b.counter.2__AT0 .def_303)))
(let ((.def_298 (and .def_6 .def_47)))
(let ((.def_299 (or b.counter.1__AT0 .def_298)))
(let ((.def_300 (and .def_286 .def_299)))
(let ((.def_301 (and b.counter.2__AT1 .def_300)))
(let ((.def_302 (or .def_50 .def_301)))
(let ((.def_305 (and .def_302 .def_304)))
(let ((.def_306 (and b.counter.3__AT1 .def_305)))
(let ((.def_307 (or b.counter.3__AT0 .def_306)))
(let ((.def_293 (and b.counter.2__AT1 .def_292)))
(let ((.def_294 (or b.counter.2__AT0 .def_293)))
(let ((.def_282 (or b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_280 (and .def_6 b.counter.0__AT1)))
(let ((.def_281 (or .def_45 .def_280)))
(let ((.def_283 (and .def_281 .def_282)))
(let ((.def_284 (and .def_4 .def_283)))
(let ((.def_285 (or .def_50 .def_284)))
(let ((.def_295 (and .def_285 .def_294)))
(let ((.def_296 (and .def_14 .def_295)))
(let ((.def_297 (or .def_53 .def_296)))
(let ((.def_308 (and .def_297 .def_307)))
(let ((.def_269 (* b.speed_x__AT0 b.speed_x__AT0)))
(let ((.def_101 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_270 (+ .def_101 .def_269)))
(let ((.def_255 (* (- 1.0) b.speed_loss__AT0)))
(let ((.def_256 (+ 1.0 .def_255)))
(let ((.def_268 (* .def_256 .def_256)))
(let ((.def_271 (* .def_268 .def_270)))
(let ((.def_272 (* (- 1.0) .def_271)))
(let ((.def_276 (+ .def_272 .def_274)))
(let ((.def_267 (* b.speed_x__AT1 b.speed_x__AT1)))
(let ((.def_277 (+ .def_267 .def_276)))
(let ((.def_278 (= .def_277 0.0 )))
(let ((.def_245 (* 2.0 b.y__AT0)))
(let ((.def_259 (* .def_245 .def_256)))
(let ((.def_260 (* b.speed_y__AT0 .def_259)))
(let ((.def_257 (* b.speed_x__AT0 .def_256)))
(let ((.def_263 (+ .def_257 .def_260)))
(let ((.def_253 (* 2.0 b.y__AT1)))
(let ((.def_254 (* b.speed_y__AT1 .def_253)))
(let ((.def_264 (+ .def_254 .def_263)))
(let ((.def_265 (+ b.speed_x__AT1 .def_264)))
(let ((.def_266 (= .def_265 0.0 )))
(let ((.def_279 (and .def_266 .def_278)))
(let ((.def_309 (and .def_279 .def_308)))
(let ((.def_252 (not b.bool_atom__AT0)))
(let ((.def_310 (or .def_252 .def_309)))
(let ((.def_246 (* b.speed_y__AT0 .def_245)))
(let ((.def_247 (+ b.speed_x__AT0 .def_246)))
(let ((.def_248 (<= 0.0 .def_247)))
(let ((.def_249 (not .def_248)))
(let ((.def_69 (* b.y__AT0 b.y__AT0)))
(let ((.def_70 (+ b.x__AT0 .def_69)))
(let ((.def_244 (= .def_70 0.0 )))
(let ((.def_250 (and .def_244 .def_249)))
(let ((.def_251 (= b.bool_atom__AT0 .def_250)))
(let ((.def_311 (and .def_251 .def_310)))
(let ((.def_315 (and .def_311 .def_314)))
(let ((.def_316 (and .def_218 .def_315)))
(let ((.def_317 (and .def_221 .def_316)))
(let ((.def_318 (or .def_197 .def_317)))
(let ((.def_319 (or b.EVENT.1__AT0 .def_318)))
(let ((.def_328 (and .def_319 .def_327)))
(let ((.def_355 (and .def_328 .def_354)))
(let ((.def_204 (= b.time__AT0 b.time__AT1)))
(let ((.def_219 (and .def_204 .def_218)))
(let ((.def_222 (and .def_219 .def_221)))
(let ((.def_225 (and .def_222 .def_224)))
(let ((.def_228 (and .def_225 .def_227)))
(let ((.def_236 (and .def_228 .def_235)))
(let ((.def_240 (and .def_236 .def_239)))
(let ((.def_241 (or .def_64 .def_240)))
(let ((.def_208 (* (- 1.0) b.time__AT1)))
(let ((.def_211 (+ b.delta__AT0 .def_208)))
(let ((.def_212 (+ b.time__AT0 .def_211)))
(let ((.def_213 (= .def_212 0.0 )))
(let ((.def_214 (or .def_199 .def_213)))
(let ((.def_215 (or b.EVENT.1__AT0 .def_214)))
(let ((.def_205 (or .def_197 .def_204)))
(let ((.def_206 (or b.EVENT.1__AT0 .def_205)))
(let ((.def_216 (and .def_206 .def_215)))
(let ((.def_242 (and .def_216 .def_241)))
(let ((.def_201 (= .def_199 b.event_is_timed__AT1)))
(let ((.def_198 (= b.event_is_timed__AT0 .def_197)))
(let ((.def_202 (and .def_198 .def_201)))
(let ((.def_243 (and .def_202 .def_242)))
(let ((.def_356 (and .def_243 .def_355)))
(let ((.def_360 (and .def_356 .def_359)))
(let ((.def_364 (and .def_360 .def_363)))
(let ((.def_365 (and .def_64 .def_364)))
(let ((.def_384 (and .def_365 .def_383)))
(let ((.def_395 (and .def_384 .def_394)))
(let ((.def_188 (+ .def_186 .def_184)))
(let ((.def_191 (<= 0.0 .def_188)))
(let ((.def_109 (* b.y__AT0 b.speed_y__AT0)))
(let ((.def_152 (* 2.0 .def_109)))
(let ((.def_153 (+ b.speed_x__AT0 .def_152)))
(let ((.def_157 (<= .def_153 0.0 )))
(let ((.def_192 (and .def_157 .def_191)))
(let ((.def_189 (<= .def_188 0.0 )))
(let ((.def_154 (<= 0.0 .def_153)))
(let ((.def_190 (and .def_154 .def_189)))
(let ((.def_193 (or .def_190 .def_192)))
(let ((.def_96 (* (- (/ 49 10)) b.speed_y__AT0)))
(let ((.def_128 (* 3.0 .def_96)))
(let ((.def_164 (* 2.0 .def_128)))
(let ((.def_165 (* b.delta__AT0 .def_164)))
(let ((.def_167 (* (- 50.0) .def_165)))
(let ((.def_169 (* (- 50.0) .def_101)))
(let ((.def_174 (+ .def_169 .def_167)))
(let ((.def_163 (* (- 7203.0) .def_87)))
(let ((.def_175 (+ .def_163 .def_174)))
(let ((.def_172 (* 490.0 b.y__AT0)))
(let ((.def_176 (+ .def_172 .def_175)))
(let ((.def_179 (<= 0.0 .def_176)))
(let ((.def_180 (and .def_157 .def_179)))
(let ((.def_177 (<= .def_176 0.0 )))
(let ((.def_178 (and .def_154 .def_177)))
(let ((.def_181 (or .def_178 .def_180)))
(let ((.def_105 (* (- (/ 49 10)) b.y__AT0)))
(let ((.def_138 (* 2.0 .def_105)))
(let ((.def_139 (* b.delta__AT0 .def_138)))
(let ((.def_140 (* 50.0 .def_139)))
(let ((.def_133 (* 2.0 .def_101)))
(let ((.def_134 (* b.delta__AT0 .def_133)))
(let ((.def_136 (* 25.0 .def_134)))
(let ((.def_146 (+ .def_136 .def_140)))
(let ((.def_129 (* .def_87 .def_128)))
(let ((.def_131 (* 50.0 .def_129)))
(let ((.def_147 (+ .def_131 .def_146)))
(let ((.def_142 (* 50.0 .def_109)))
(let ((.def_148 (+ .def_142 .def_147)))
(let ((.def_88 (* b.delta__AT0 .def_87)))
(let ((.def_126 (* 2401.0 .def_88)))
(let ((.def_149 (+ .def_126 .def_148)))
(let ((.def_144 (* 25.0 b.speed_x__AT0)))
(let ((.def_150 (+ .def_144 .def_149)))
(let ((.def_156 (<= .def_150 0.0 )))
(let ((.def_158 (and .def_156 .def_157)))
(let ((.def_151 (<= 0.0 .def_150)))
(let ((.def_155 (and .def_151 .def_154)))
(let ((.def_159 (or .def_155 .def_158)))
(let ((.def_110 (* b.delta__AT0 .def_109)))
(let ((.def_111 (* 200.0 .def_110)))
(let ((.def_106 (* .def_87 .def_105)))
(let ((.def_107 (* 200.0 .def_106)))
(let ((.def_117 (+ .def_107 .def_111)))
(let ((.def_102 (* .def_87 .def_101)))
(let ((.def_103 (* 100.0 .def_102)))
(let ((.def_118 (+ .def_103 .def_117)))
(let ((.def_97 (* .def_88 .def_96)))
(let ((.def_99 (* 200.0 .def_97)))
(let ((.def_119 (+ .def_99 .def_118)))
(let ((.def_89 (* b.delta__AT0 .def_88)))
(let ((.def_91 (* 2401.0 .def_89)))
(let ((.def_120 (+ .def_91 .def_119)))
(let ((.def_86 (* 100.0 .def_84)))
(let ((.def_121 (+ .def_86 .def_120)))
(let ((.def_113 (* 100.0 .def_69)))
(let ((.def_122 (+ .def_113 .def_121)))
(let ((.def_115 (* 100.0 b.x__AT0)))
(let ((.def_123 (+ .def_115 .def_122)))
(let ((.def_124 (<= 0.0 .def_123)))
(let ((.def_71 (<= 0.0 .def_70)))
(let ((.def_125 (and .def_71 .def_124)))
(let ((.def_160 (and .def_125 .def_159)))
(let ((.def_182 (and .def_160 .def_181)))
(let ((.def_194 (and .def_182 .def_193)))
(let ((.def_195 (and .def_78 .def_194)))
(let ((.def_79 (and .def_71 .def_78)))
(let ((.def_67 (or .def_64 .def_66)))
(let ((.def_57 (or .def_45 .def_47)))
(let ((.def_61 (or b.counter.3__AT0 .def_57)))
(let ((.def_58 (or b.counter.2__AT0 .def_57)))
(let ((.def_56 (or .def_47 .def_50)))
(let ((.def_59 (and .def_56 .def_58)))
(let ((.def_60 (or .def_53 .def_59)))
(let ((.def_62 (and .def_60 .def_61)))
(let ((.def_68 (and .def_62 .def_67)))
(let ((.def_80 (and .def_68 .def_79)))
(let ((.def_51 (and .def_48 .def_50)))
(let ((.def_54 (and .def_51 .def_53)))
(let ((.def_42 (= b.speed_y__AT0 1.0 )))
(let ((.def_39 (= b.speed_x__AT0 1.0 )))
(let ((.def_35 (= b.y__AT0 10.0 )))
(let ((.def_31 (= b.x__AT0 (- 9.0) )))
(let ((.def_25 (= b.time__AT0 0.0 )))
(let ((.def_27 (and .def_25 b.event_is_timed__AT0)))
(let ((.def_32 (and .def_27 .def_31)))
(let ((.def_36 (and .def_32 .def_35)))
(let ((.def_40 (and .def_36 .def_39)))
(let ((.def_43 (and .def_40 .def_42)))
(let ((.def_55 (and .def_43 .def_54)))
(let ((.def_81 (and .def_55 .def_80)))
(let ((.def_17 (or b.counter.1__AT1 .def_9)))
(let ((.def_18 (or b.counter.2__AT1 .def_17)))
(let ((.def_10 (and .def_6 .def_9)))
(let ((.def_16 (or .def_4 .def_10)))
(let ((.def_19 (and .def_16 .def_18)))
(let ((.def_20 (or b.counter.3__AT1 .def_19)))
(let ((.def_11 (or b.counter.2__AT1 .def_10)))
(let ((.def_7 (or .def_4 .def_6)))
(let ((.def_12 (and .def_7 .def_11)))
(let ((.def_15 (or .def_12 .def_14)))
(let ((.def_21 (and .def_15 .def_20)))
(let ((.def_22 (not .def_21)))
(let ((.def_82 (and .def_22 .def_81)))
(let ((.def_196 (and .def_82 .def_195)))
(let ((.def_396 (and .def_196 .def_395)))
(let ((.def_493 (and .def_396 .def_492)))
.def_493))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
