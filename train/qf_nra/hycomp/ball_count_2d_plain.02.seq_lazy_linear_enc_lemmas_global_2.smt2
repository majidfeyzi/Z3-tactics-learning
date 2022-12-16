(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 2 and uses the quantifier free encoding with equivalences and lemmas encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:47:59 2012
(declare-fun b.time__AT1 () Real)
(declare-fun b.speed_x__AT2 () Real)
(declare-fun g__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.speed_y__AT1 () Real)
(declare-fun b.delta__AT1 () Real)
(declare-fun b.speed_y__AT2 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.x__AT1 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.event_is_timed__AT2 () Bool)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.y__AT1 () Real)
(declare-fun b.time__AT2 () Real)
(declare-fun b.y__AT2 () Real)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.speed_x__AT0 () Real)
(declare-fun b.delta__AT2 () Real)
(declare-fun b.counter.0__AT2 () Bool)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.x__AT2 () Real)
(declare-fun b.EVENT.0__AT2 () Bool)
(declare-fun b.counter.1__AT2 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.EVENT.1__AT2 () Bool)
(declare-fun b.counter.2__AT2 () Bool)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.counter.3__AT2 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.x__AT0 () Real)
(declare-fun b.speed_x__AT1 () Real)
(assert (let ((.def_71 (* (- 1.0) g__AT0)))
(let ((.def_72 (* (/ 1 2) .def_71)))
(let ((.def_79 (* 2.0 .def_72)))
(let ((.def_502 (* .def_79 b.delta__AT2)))
(let ((.def_503 (+ b.speed_y__AT2 .def_502)))
(let ((.def_520 (<= .def_503 0.0 )))
(let ((.def_521 (not .def_520)))
(let ((.def_518 (<= b.speed_y__AT2 0.0 )))
(let ((.def_541 (or .def_518 .def_521)))
(let ((.def_506 (<= 0.0 b.speed_y__AT2)))
(let ((.def_507 (not .def_506)))
(let ((.def_504 (<= 0.0 .def_503)))
(let ((.def_540 (or .def_504 .def_507)))
(let ((.def_542 (and .def_540 .def_541)))
(let ((.def_119 (<= 0.0 g__AT0)))
(let ((.def_543 (or .def_119 .def_542)))
(let ((.def_505 (not .def_504)))
(let ((.def_537 (or .def_505 .def_506)))
(let ((.def_519 (not .def_518)))
(let ((.def_536 (or .def_519 .def_520)))
(let ((.def_538 (and .def_536 .def_537)))
(let ((.def_114 (<= g__AT0 0.0 )))
(let ((.def_539 (or .def_114 .def_538)))
(let ((.def_544 (and .def_539 .def_543)))
(let ((.def_533 (and .def_518 .def_520)))
(let ((.def_532 (and .def_504 .def_506)))
(let ((.def_534 (or .def_532 .def_533)))
(let ((.def_497 (* b.speed_y__AT2 b.delta__AT2)))
(let ((.def_495 (* b.delta__AT2 b.delta__AT2)))
(let ((.def_496 (* .def_72 .def_495)))
(let ((.def_498 (+ .def_496 .def_497)))
(let ((.def_499 (+ b.y__AT2 .def_498)))
(let ((.def_512 (<= .def_499 0.0 )))
(let ((.def_526 (not .def_512)))
(let ((.def_510 (<= b.y__AT2 0.0 )))
(let ((.def_527 (or .def_510 .def_526)))
(let ((.def_486 (<= 0.0 b.y__AT2)))
(let ((.def_524 (not .def_486)))
(let ((.def_500 (<= 0.0 .def_499)))
(let ((.def_525 (or .def_500 .def_524)))
(let ((.def_528 (and .def_525 .def_527)))
(let ((.def_522 (or .def_519 .def_521)))
(let ((.def_523 (not .def_522)))
(let ((.def_529 (or .def_523 .def_528)))
(let ((.def_514 (not .def_500)))
(let ((.def_515 (or .def_486 .def_514)))
(let ((.def_511 (not .def_510)))
(let ((.def_513 (or .def_511 .def_512)))
(let ((.def_516 (and .def_513 .def_515)))
(let ((.def_508 (or .def_505 .def_507)))
(let ((.def_509 (not .def_508)))
(let ((.def_517 (or .def_509 .def_516)))
(let ((.def_530 (and .def_517 .def_529)))
(let ((.def_501 (and .def_486 .def_500)))
(let ((.def_531 (and .def_501 .def_530)))
(let ((.def_535 (and .def_531 .def_534)))
(let ((.def_545 (and .def_535 .def_544)))
(let ((.def_52 (<= g__AT0 10.0 )))
(let ((.def_51 (<= 8.0 g__AT0)))
(let ((.def_53 (and .def_51 .def_52)))
(let ((.def_546 (and .def_53 .def_545)))
(let ((.def_60 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_57 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_61 (and .def_57 .def_60)))
(let ((.def_547 (and .def_61 .def_546)))
(let ((.def_182 (not b.counter.0__AT1)))
(let ((.def_170 (not b.counter.1__AT1)))
(let ((.def_490 (and .def_170 .def_182)))
(let ((.def_177 (not b.counter.2__AT1)))
(let ((.def_491 (and .def_177 .def_490)))
(let ((.def_62 (and .def_53 .def_61)))
(let ((.def_487 (and .def_62 .def_486)))
(let ((.def_483 (not b.EVENT.0__AT2)))
(let ((.def_481 (not b.EVENT.1__AT2)))
(let ((.def_484 (or .def_481 .def_483)))
(let ((.def_6 (not b.counter.0__AT2)))
(let ((.def_4 (not b.counter.1__AT2)))
(let ((.def_474 (or .def_4 .def_6)))
(let ((.def_478 (or b.counter.3__AT2 .def_474)))
(let ((.def_475 (or b.counter.2__AT2 .def_474)))
(let ((.def_9 (not b.counter.2__AT2)))
(let ((.def_473 (or .def_6 .def_9)))
(let ((.def_476 (and .def_473 .def_475)))
(let ((.def_395 (not b.counter.3__AT2)))
(let ((.def_477 (or .def_395 .def_476)))
(let ((.def_479 (and .def_477 .def_478)))
(let ((.def_485 (and .def_479 .def_484)))
(let ((.def_488 (and .def_485 .def_487)))
(let ((.def_468 (<= 0.0 b.delta__AT1)))
(let ((.def_280 (not b.EVENT.0__AT1)))
(let ((.def_278 (not b.EVENT.1__AT1)))
(let ((.def_370 (and .def_278 .def_280)))
(let ((.def_371 (not .def_370)))
(let ((.def_469 (or .def_371 .def_468)))
(let ((.def_470 (or b.EVENT.1__AT1 .def_469)))
(let ((.def_417 (= b.counter.0__AT2 b.counter.0__AT1)))
(let ((.def_415 (= b.counter.1__AT2 b.counter.1__AT1)))
(let ((.def_413 (= b.counter.2__AT2 b.counter.2__AT1)))
(let ((.def_412 (= b.counter.3__AT1 b.counter.3__AT2)))
(let ((.def_414 (and .def_412 .def_413)))
(let ((.def_416 (and .def_414 .def_415)))
(let ((.def_418 (and .def_416 .def_417)))
(let ((.def_465 (or .def_371 .def_418)))
(let ((.def_466 (or b.EVENT.1__AT1 .def_465)))
(let ((.def_425 (= b.x__AT1 b.x__AT2)))
(let ((.def_422 (= b.y__AT1 b.y__AT2)))
(let ((.def_459 (and .def_422 .def_425)))
(let ((.def_351 (= b.speed_x__AT1 b.speed_x__AT2)))
(let ((.def_460 (and .def_351 .def_459)))
(let ((.def_411 (= b.speed_y__AT1 b.speed_y__AT2)))
(let ((.def_461 (and .def_411 .def_460)))
(let ((.def_456 (= b.delta__AT1 0.0 )))
(let ((.def_457 (and .def_370 .def_456)))
(let ((.def_458 (not .def_457)))
(let ((.def_462 (or .def_458 .def_461)))
(let ((.def_463 (or b.EVENT.1__AT1 .def_462)))
(let ((.def_436 (= b.time__AT1 b.time__AT2)))
(let ((.def_448 (and .def_425 .def_436)))
(let ((.def_449 (and .def_422 .def_448)))
(let ((.def_450 (and .def_351 .def_449)))
(let ((.def_451 (and .def_411 .def_450)))
(let ((.def_452 (and .def_418 .def_451)))
(let ((.def_453 (or .def_278 .def_452)))
(let ((.def_439 (* (- 1.0) b.time__AT2)))
(let ((.def_442 (+ b.delta__AT1 .def_439)))
(let ((.def_443 (+ b.time__AT1 .def_442)))
(let ((.def_444 (= .def_443 0.0 )))
(let ((.def_445 (or .def_371 .def_444)))
(let ((.def_446 (or b.EVENT.1__AT1 .def_445)))
(let ((.def_437 (or .def_370 .def_436)))
(let ((.def_438 (or b.EVENT.1__AT1 .def_437)))
(let ((.def_447 (and .def_438 .def_446)))
(let ((.def_454 (and .def_447 .def_453)))
(let ((.def_432 (= .def_371 b.event_is_timed__AT2)))
(let ((.def_430 (= b.event_is_timed__AT1 .def_370)))
(let ((.def_433 (and .def_430 .def_432)))
(let ((.def_419 (and .def_411 .def_418)))
(let ((.def_373 (= b.y__AT1 0.0 )))
(let ((.def_301 (<= 0.0 b.speed_y__AT1)))
(let ((.def_302 (not .def_301)))
(let ((.def_374 (and .def_302 .def_373)))
(let ((.def_420 (or .def_374 .def_419)))
(let ((.def_386 (or .def_6 b.counter.0__AT1)))
(let ((.def_385 (or b.counter.0__AT2 .def_182)))
(let ((.def_387 (and .def_385 .def_386)))
(let ((.def_388 (and .def_4 .def_387)))
(let ((.def_389 (or b.counter.1__AT1 .def_388)))
(let ((.def_384 (or b.counter.1__AT2 .def_170)))
(let ((.def_390 (and .def_384 .def_389)))
(let ((.def_403 (and .def_9 .def_390)))
(let ((.def_404 (or b.counter.2__AT1 .def_403)))
(let ((.def_398 (and .def_4 .def_182)))
(let ((.def_399 (or b.counter.1__AT1 .def_398)))
(let ((.def_400 (and .def_384 .def_399)))
(let ((.def_401 (and b.counter.2__AT2 .def_400)))
(let ((.def_402 (or .def_177 .def_401)))
(let ((.def_405 (and .def_402 .def_404)))
(let ((.def_406 (and b.counter.3__AT2 .def_405)))
(let ((.def_407 (or b.counter.3__AT1 .def_406)))
(let ((.def_391 (and b.counter.2__AT2 .def_390)))
(let ((.def_392 (or b.counter.2__AT1 .def_391)))
(let ((.def_380 (or b.counter.1__AT2 b.counter.1__AT1)))
(let ((.def_378 (and .def_4 b.counter.0__AT2)))
(let ((.def_379 (or .def_170 .def_378)))
(let ((.def_381 (and .def_379 .def_380)))
(let ((.def_382 (and .def_9 .def_381)))
(let ((.def_383 (or .def_177 .def_382)))
(let ((.def_393 (and .def_383 .def_392)))
(let ((.def_396 (and .def_393 .def_395)))
(let ((.def_192 (not b.counter.3__AT1)))
(let ((.def_397 (or .def_192 .def_396)))
(let ((.def_408 (and .def_397 .def_407)))
(let ((.def_165 (* (- 1.0) speed_loss__AT0)))
(let ((.def_166 (+ 1.0 .def_165)))
(let ((.def_144 (* (- 1.0) b.speed_y__AT1)))
(let ((.def_376 (* .def_144 .def_166)))
(let ((.def_377 (= b.speed_y__AT2 .def_376)))
(let ((.def_409 (and .def_377 .def_408)))
(let ((.def_375 (not .def_374)))
(let ((.def_410 (or .def_375 .def_409)))
(let ((.def_421 (and .def_410 .def_420)))
(let ((.def_423 (and .def_421 .def_422)))
(let ((.def_424 (and .def_351 .def_423)))
(let ((.def_426 (and .def_424 .def_425)))
(let ((.def_427 (or .def_370 .def_426)))
(let ((.def_428 (or b.EVENT.1__AT1 .def_427)))
(let ((.def_361 (* (- 1.0) b.y__AT2)))
(let ((.def_292 (* b.speed_y__AT1 b.delta__AT1)))
(let ((.def_365 (+ .def_292 .def_361)))
(let ((.def_290 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_291 (* .def_72 .def_290)))
(let ((.def_366 (+ .def_291 .def_365)))
(let ((.def_367 (+ b.y__AT1 .def_366)))
(let ((.def_368 (= .def_367 0.0 )))
(let ((.def_355 (* (- 1.0) b.speed_y__AT2)))
(let ((.def_353 (* .def_71 b.delta__AT1)))
(let ((.def_356 (+ .def_353 .def_355)))
(let ((.def_357 (+ b.speed_y__AT1 .def_356)))
(let ((.def_358 (= .def_357 0.0 )))
(let ((.def_346 (* (- 1.0) b.x__AT2)))
(let ((.def_344 (* b.speed_x__AT1 b.delta__AT1)))
(let ((.def_347 (+ .def_344 .def_346)))
(let ((.def_348 (+ b.x__AT1 .def_347)))
(let ((.def_349 (= .def_348 0.0 )))
(let ((.def_352 (and .def_349 .def_351)))
(let ((.def_359 (and .def_352 .def_358)))
(let ((.def_369 (and .def_359 .def_368)))
(let ((.def_372 (or .def_369 .def_371)))
(let ((.def_429 (and .def_372 .def_428)))
(let ((.def_434 (and .def_429 .def_433)))
(let ((.def_455 (and .def_434 .def_454)))
(let ((.def_464 (and .def_455 .def_463)))
(let ((.def_467 (and .def_464 .def_466)))
(let ((.def_471 (and .def_467 .def_470)))
(let ((.def_472 (and .def_278 .def_471)))
(let ((.def_489 (and .def_472 .def_488)))
(let ((.def_492 (and .def_489 .def_491)))
(let ((.def_297 (* .def_79 b.delta__AT1)))
(let ((.def_298 (+ b.speed_y__AT1 .def_297)))
(let ((.def_315 (<= .def_298 0.0 )))
(let ((.def_316 (not .def_315)))
(let ((.def_313 (<= b.speed_y__AT1 0.0 )))
(let ((.def_336 (or .def_313 .def_316)))
(let ((.def_299 (<= 0.0 .def_298)))
(let ((.def_335 (or .def_299 .def_302)))
(let ((.def_337 (and .def_335 .def_336)))
(let ((.def_338 (or .def_119 .def_337)))
(let ((.def_300 (not .def_299)))
(let ((.def_332 (or .def_300 .def_301)))
(let ((.def_314 (not .def_313)))
(let ((.def_331 (or .def_314 .def_315)))
(let ((.def_333 (and .def_331 .def_332)))
(let ((.def_334 (or .def_114 .def_333)))
(let ((.def_339 (and .def_334 .def_338)))
(let ((.def_328 (and .def_313 .def_315)))
(let ((.def_327 (and .def_299 .def_301)))
(let ((.def_329 (or .def_327 .def_328)))
(let ((.def_293 (+ .def_291 .def_292)))
(let ((.def_294 (+ b.y__AT1 .def_293)))
(let ((.def_307 (<= .def_294 0.0 )))
(let ((.def_321 (not .def_307)))
(let ((.def_305 (<= b.y__AT1 0.0 )))
(let ((.def_322 (or .def_305 .def_321)))
(let ((.def_283 (<= 0.0 b.y__AT1)))
(let ((.def_319 (not .def_283)))
(let ((.def_295 (<= 0.0 .def_294)))
(let ((.def_320 (or .def_295 .def_319)))
(let ((.def_323 (and .def_320 .def_322)))
(let ((.def_317 (or .def_314 .def_316)))
(let ((.def_318 (not .def_317)))
(let ((.def_324 (or .def_318 .def_323)))
(let ((.def_309 (not .def_295)))
(let ((.def_310 (or .def_283 .def_309)))
(let ((.def_306 (not .def_305)))
(let ((.def_308 (or .def_306 .def_307)))
(let ((.def_311 (and .def_308 .def_310)))
(let ((.def_303 (or .def_300 .def_302)))
(let ((.def_304 (not .def_303)))
(let ((.def_312 (or .def_304 .def_311)))
(let ((.def_325 (and .def_312 .def_324)))
(let ((.def_296 (and .def_283 .def_295)))
(let ((.def_326 (and .def_296 .def_325)))
(let ((.def_330 (and .def_326 .def_329)))
(let ((.def_340 (and .def_330 .def_339)))
(let ((.def_341 (and .def_53 .def_340)))
(let ((.def_342 (and .def_61 .def_341)))
(let ((.def_284 (and .def_62 .def_283)))
(let ((.def_281 (or .def_278 .def_280)))
(let ((.def_271 (or .def_170 .def_182)))
(let ((.def_275 (or b.counter.3__AT1 .def_271)))
(let ((.def_272 (or b.counter.2__AT1 .def_271)))
(let ((.def_270 (or .def_177 .def_182)))
(let ((.def_273 (and .def_270 .def_272)))
(let ((.def_274 (or .def_192 .def_273)))
(let ((.def_276 (and .def_274 .def_275)))
(let ((.def_282 (and .def_276 .def_281)))
(let ((.def_285 (and .def_282 .def_284)))
(let ((.def_265 (<= 0.0 b.delta__AT0)))
(let ((.def_46 (not b.EVENT.0__AT0)))
(let ((.def_44 (not b.EVENT.1__AT0)))
(let ((.def_158 (and .def_44 .def_46)))
(let ((.def_159 (not .def_158)))
(let ((.def_266 (or .def_159 .def_265)))
(let ((.def_267 (or b.EVENT.1__AT0 .def_266)))
(let ((.def_214 (= b.counter.0__AT0 b.counter.0__AT1)))
(let ((.def_212 (= b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_210 (= b.counter.2__AT0 b.counter.2__AT1)))
(let ((.def_209 (= b.counter.3__AT0 b.counter.3__AT1)))
(let ((.def_211 (and .def_209 .def_210)))
(let ((.def_213 (and .def_211 .def_212)))
(let ((.def_215 (and .def_213 .def_214)))
(let ((.def_262 (or .def_159 .def_215)))
(let ((.def_263 (or b.EVENT.1__AT0 .def_262)))
(let ((.def_222 (= b.x__AT1 b.x__AT0)))
(let ((.def_219 (= b.y__AT0 b.y__AT1)))
(let ((.def_256 (and .def_219 .def_222)))
(let ((.def_140 (= b.speed_x__AT0 b.speed_x__AT1)))
(let ((.def_257 (and .def_140 .def_256)))
(let ((.def_208 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_258 (and .def_208 .def_257)))
(let ((.def_253 (= b.delta__AT0 0.0 )))
(let ((.def_254 (and .def_158 .def_253)))
(let ((.def_255 (not .def_254)))
(let ((.def_259 (or .def_255 .def_258)))
(let ((.def_260 (or b.EVENT.1__AT0 .def_259)))
(let ((.def_233 (= b.time__AT0 b.time__AT1)))
(let ((.def_245 (and .def_222 .def_233)))
(let ((.def_246 (and .def_219 .def_245)))
(let ((.def_247 (and .def_140 .def_246)))
(let ((.def_248 (and .def_208 .def_247)))
(let ((.def_249 (and .def_215 .def_248)))
(let ((.def_250 (or .def_44 .def_249)))
(let ((.def_236 (* (- 1.0) b.time__AT1)))
(let ((.def_239 (+ b.delta__AT0 .def_236)))
(let ((.def_240 (+ b.time__AT0 .def_239)))
(let ((.def_241 (= .def_240 0.0 )))
(let ((.def_242 (or .def_159 .def_241)))
(let ((.def_243 (or b.EVENT.1__AT0 .def_242)))
(let ((.def_234 (or .def_158 .def_233)))
(let ((.def_235 (or b.EVENT.1__AT0 .def_234)))
(let ((.def_244 (and .def_235 .def_243)))
(let ((.def_251 (and .def_244 .def_250)))
(let ((.def_229 (= .def_159 b.event_is_timed__AT1)))
(let ((.def_227 (= b.event_is_timed__AT0 .def_158)))
(let ((.def_230 (and .def_227 .def_229)))
(let ((.def_216 (and .def_208 .def_215)))
(let ((.def_161 (= b.y__AT0 0.0 )))
(let ((.def_84 (<= 0.0 b.speed_y__AT0)))
(let ((.def_85 (not .def_84)))
(let ((.def_162 (and .def_85 .def_161)))
(let ((.def_217 (or .def_162 .def_216)))
(let ((.def_183 (or b.counter.0__AT0 .def_182)))
(let ((.def_27 (not b.counter.0__AT0)))
(let ((.def_181 (or .def_27 b.counter.0__AT1)))
(let ((.def_184 (and .def_181 .def_183)))
(let ((.def_185 (and .def_170 .def_184)))
(let ((.def_186 (or b.counter.1__AT0 .def_185)))
(let ((.def_25 (not b.counter.1__AT0)))
(let ((.def_180 (or .def_25 b.counter.1__AT1)))
(let ((.def_187 (and .def_180 .def_186)))
(let ((.def_200 (and .def_177 .def_187)))
(let ((.def_201 (or b.counter.2__AT0 .def_200)))
(let ((.def_195 (and .def_27 .def_170)))
(let ((.def_196 (or b.counter.1__AT0 .def_195)))
(let ((.def_197 (and .def_180 .def_196)))
(let ((.def_198 (and b.counter.2__AT1 .def_197)))
(let ((.def_30 (not b.counter.2__AT0)))
(let ((.def_199 (or .def_30 .def_198)))
(let ((.def_202 (and .def_199 .def_201)))
(let ((.def_203 (and b.counter.3__AT1 .def_202)))
(let ((.def_204 (or b.counter.3__AT0 .def_203)))
(let ((.def_188 (and b.counter.2__AT1 .def_187)))
(let ((.def_189 (or b.counter.2__AT0 .def_188)))
(let ((.def_174 (or b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_172 (and .def_170 b.counter.0__AT1)))
(let ((.def_173 (or .def_25 .def_172)))
(let ((.def_175 (and .def_173 .def_174)))
(let ((.def_178 (and .def_175 .def_177)))
(let ((.def_179 (or .def_30 .def_178)))
(let ((.def_190 (and .def_179 .def_189)))
(let ((.def_193 (and .def_190 .def_192)))
(let ((.def_33 (not b.counter.3__AT0)))
(let ((.def_194 (or .def_33 .def_193)))
(let ((.def_205 (and .def_194 .def_204)))
(let ((.def_164 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_167 (* .def_164 .def_166)))
(let ((.def_168 (= b.speed_y__AT1 .def_167)))
(let ((.def_206 (and .def_168 .def_205)))
(let ((.def_163 (not .def_162)))
(let ((.def_207 (or .def_163 .def_206)))
(let ((.def_218 (and .def_207 .def_217)))
(let ((.def_220 (and .def_218 .def_219)))
(let ((.def_221 (and .def_140 .def_220)))
(let ((.def_223 (and .def_221 .def_222)))
(let ((.def_224 (or .def_158 .def_223)))
(let ((.def_225 (or b.EVENT.1__AT0 .def_224)))
(let ((.def_150 (* (- 1.0) b.y__AT1)))
(let ((.def_74 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_153 (+ .def_74 .def_150)))
(let ((.def_69 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_73 (* .def_69 .def_72)))
(let ((.def_154 (+ .def_73 .def_153)))
(let ((.def_155 (+ b.y__AT0 .def_154)))
(let ((.def_156 (= .def_155 0.0 )))
(let ((.def_142 (* b.delta__AT0 .def_71)))
(let ((.def_145 (+ .def_142 .def_144)))
(let ((.def_146 (+ b.speed_y__AT0 .def_145)))
(let ((.def_147 (= .def_146 0.0 )))
(let ((.def_132 (* (- 1.0) b.x__AT1)))
(let ((.def_136 (+ .def_132 b.x__AT0)))
(let ((.def_130 (* b.delta__AT0 b.speed_x__AT0)))
(let ((.def_137 (+ .def_130 .def_136)))
(let ((.def_138 (= .def_137 0.0 )))
(let ((.def_141 (and .def_138 .def_140)))
(let ((.def_148 (and .def_141 .def_147)))
(let ((.def_157 (and .def_148 .def_156)))
(let ((.def_160 (or .def_157 .def_159)))
(let ((.def_226 (and .def_160 .def_225)))
(let ((.def_231 (and .def_226 .def_230)))
(let ((.def_252 (and .def_231 .def_251)))
(let ((.def_261 (and .def_252 .def_260)))
(let ((.def_264 (and .def_261 .def_263)))
(let ((.def_268 (and .def_264 .def_267)))
(let ((.def_269 (and .def_44 .def_268)))
(let ((.def_286 (and .def_269 .def_285)))
(let ((.def_28 (and .def_25 .def_27)))
(let ((.def_31 (and .def_28 .def_30)))
(let ((.def_287 (and .def_31 .def_286)))
(let ((.def_80 (* b.delta__AT0 .def_79)))
(let ((.def_81 (+ b.speed_y__AT0 .def_80)))
(let ((.def_98 (<= .def_81 0.0 )))
(let ((.def_99 (not .def_98)))
(let ((.def_96 (<= b.speed_y__AT0 0.0 )))
(let ((.def_121 (or .def_96 .def_99)))
(let ((.def_82 (<= 0.0 .def_81)))
(let ((.def_120 (or .def_82 .def_85)))
(let ((.def_122 (and .def_120 .def_121)))
(let ((.def_123 (or .def_119 .def_122)))
(let ((.def_83 (not .def_82)))
(let ((.def_116 (or .def_83 .def_84)))
(let ((.def_97 (not .def_96)))
(let ((.def_115 (or .def_97 .def_98)))
(let ((.def_117 (and .def_115 .def_116)))
(let ((.def_118 (or .def_114 .def_117)))
(let ((.def_124 (and .def_118 .def_123)))
(let ((.def_111 (and .def_96 .def_98)))
(let ((.def_110 (and .def_82 .def_84)))
(let ((.def_112 (or .def_110 .def_111)))
(let ((.def_75 (+ .def_73 .def_74)))
(let ((.def_76 (+ b.y__AT0 .def_75)))
(let ((.def_90 (<= .def_76 0.0 )))
(let ((.def_104 (not .def_90)))
(let ((.def_88 (<= b.y__AT0 0.0 )))
(let ((.def_105 (or .def_88 .def_104)))
(let ((.def_63 (<= 0.0 b.y__AT0)))
(let ((.def_102 (not .def_63)))
(let ((.def_77 (<= 0.0 .def_76)))
(let ((.def_103 (or .def_77 .def_102)))
(let ((.def_106 (and .def_103 .def_105)))
(let ((.def_100 (or .def_97 .def_99)))
(let ((.def_101 (not .def_100)))
(let ((.def_107 (or .def_101 .def_106)))
(let ((.def_92 (not .def_77)))
(let ((.def_93 (or .def_63 .def_92)))
(let ((.def_89 (not .def_88)))
(let ((.def_91 (or .def_89 .def_90)))
(let ((.def_94 (and .def_91 .def_93)))
(let ((.def_86 (or .def_83 .def_85)))
(let ((.def_87 (not .def_86)))
(let ((.def_95 (or .def_87 .def_94)))
(let ((.def_108 (and .def_95 .def_107)))
(let ((.def_78 (and .def_63 .def_77)))
(let ((.def_109 (and .def_78 .def_108)))
(let ((.def_113 (and .def_109 .def_112)))
(let ((.def_125 (and .def_113 .def_124)))
(let ((.def_126 (and .def_53 .def_125)))
(let ((.def_127 (and .def_61 .def_126)))
(let ((.def_64 (and .def_62 .def_63)))
(let ((.def_47 (or .def_44 .def_46)))
(let ((.def_37 (or .def_25 .def_27)))
(let ((.def_41 (or b.counter.3__AT0 .def_37)))
(let ((.def_38 (or b.counter.2__AT0 .def_37)))
(let ((.def_36 (or .def_27 .def_30)))
(let ((.def_39 (and .def_36 .def_38)))
(let ((.def_40 (or .def_33 .def_39)))
(let ((.def_42 (and .def_40 .def_41)))
(let ((.def_48 (and .def_42 .def_47)))
(let ((.def_65 (and .def_48 .def_64)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_22 (= b.speed_y__AT0 0.0 )))
(let ((.def_19 (= b.y__AT0 10.0 )))
(let ((.def_14 (= b.time__AT0 0.0 )))
(let ((.def_16 (and .def_14 b.event_is_timed__AT0)))
(let ((.def_20 (and .def_16 .def_19)))
(let ((.def_23 (and .def_20 .def_22)))
(let ((.def_35 (and .def_23 .def_34)))
(let ((.def_66 (and .def_35 .def_65)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_11 (not .def_10)))
(let ((.def_67 (and .def_11 .def_66)))
(let ((.def_128 (and .def_67 .def_127)))
(let ((.def_288 (and .def_128 .def_287)))
(let ((.def_343 (and .def_288 .def_342)))
(let ((.def_493 (and .def_343 .def_492)))
(let ((.def_548 (and .def_493 .def_547)))
.def_548)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
