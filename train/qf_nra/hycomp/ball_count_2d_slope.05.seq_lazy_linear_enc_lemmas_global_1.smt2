(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 1 and uses the quantifier free encoding with equivalences and lemmas encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:43:18 2012
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
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.speed_x__AT1 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.time__AT1 () Real)
(declare-fun b.x__AT1 () Real)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.speed_loss__AT0 () Real)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.speed_x__AT0 () Real)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.bool_atom__AT1 () Bool)
(assert (let ((.def_602 (* (- 49.0) b.delta__AT1)))
(let ((.def_603 (* 5.0 b.speed_y__AT1)))
(let ((.def_605 (+ .def_603 .def_602)))
(let ((.def_618 (<= 0.0 .def_605)))
(let ((.def_619 (not .def_618)))
(let ((.def_616 (<= 0.0 b.speed_y__AT1)))
(let ((.def_633 (or .def_616 .def_619)))
(let ((.def_608 (<= b.speed_y__AT1 0.0 )))
(let ((.def_609 (not .def_608)))
(let ((.def_606 (<= .def_605 0.0 )))
(let ((.def_632 (or .def_606 .def_609)))
(let ((.def_634 (and .def_632 .def_633)))
(let ((.def_488 (* b.y__AT1 b.speed_y__AT1)))
(let ((.def_529 (* 2.0 .def_488)))
(let ((.def_530 (+ b.speed_x__AT1 .def_529)))
(let ((.def_543 (<= .def_530 0.0 )))
(let ((.def_629 (and .def_543 .def_618)))
(let ((.def_531 (<= 0.0 .def_530)))
(let ((.def_628 (and .def_531 .def_606)))
(let ((.def_630 (or .def_628 .def_629)))
(let ((.def_477 (* (- (/ 49 10)) b.speed_y__AT1)))
(let ((.def_506 (* 3.0 .def_477)))
(let ((.def_562 (* 2.0 .def_506)))
(let ((.def_563 (* b.delta__AT1 .def_562)))
(let ((.def_564 (* (- 50.0) .def_563)))
(let ((.def_472 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_561 (* (- 7203.0) .def_472)))
(let ((.def_565 (+ .def_561 .def_564)))
(let ((.def_347 (* b.speed_y__AT1 b.speed_y__AT1)))
(let ((.def_566 (* (- 50.0) .def_347)))
(let ((.def_570 (+ .def_566 .def_565)))
(let ((.def_568 (* 490.0 b.y__AT1)))
(let ((.def_571 (+ .def_568 .def_570)))
(let ((.def_588 (<= 0.0 .def_571)))
(let ((.def_589 (not .def_588)))
(let ((.def_574 (* (- 5.0) .def_347)))
(let ((.def_575 (* 49.0 b.y__AT1)))
(let ((.def_577 (+ .def_575 .def_574)))
(let ((.def_586 (<= 0.0 .def_577)))
(let ((.def_623 (or .def_586 .def_589)))
(let ((.def_578 (<= .def_577 0.0 )))
(let ((.def_579 (not .def_578)))
(let ((.def_572 (<= .def_571 0.0 )))
(let ((.def_622 (or .def_572 .def_579)))
(let ((.def_624 (and .def_622 .def_623)))
(let ((.def_617 (not .def_616)))
(let ((.def_620 (or .def_617 .def_619)))
(let ((.def_621 (not .def_620)))
(let ((.def_625 (or .def_621 .def_624)))
(let ((.def_573 (not .def_572)))
(let ((.def_613 (or .def_573 .def_578)))
(let ((.def_587 (not .def_586)))
(let ((.def_612 (or .def_587 .def_588)))
(let ((.def_614 (and .def_612 .def_613)))
(let ((.def_607 (not .def_606)))
(let ((.def_610 (or .def_607 .def_609)))
(let ((.def_611 (not .def_610)))
(let ((.def_615 (or .def_611 .def_614)))
(let ((.def_626 (and .def_615 .def_625)))
(let ((.def_599 (and .def_543 .def_588)))
(let ((.def_598 (and .def_531 .def_572)))
(let ((.def_600 (or .def_598 .def_599)))
(let ((.def_484 (* (- (/ 49 10)) b.y__AT1)))
(let ((.def_514 (* 2.0 .def_484)))
(let ((.def_515 (* b.delta__AT1 .def_514)))
(let ((.def_516 (* 50.0 .def_515)))
(let ((.def_510 (* 2.0 .def_347)))
(let ((.def_511 (* b.delta__AT1 .def_510)))
(let ((.def_512 (* 25.0 .def_511)))
(let ((.def_522 (+ .def_512 .def_516)))
(let ((.def_507 (* .def_472 .def_506)))
(let ((.def_508 (* 50.0 .def_507)))
(let ((.def_523 (+ .def_508 .def_522)))
(let ((.def_518 (* 50.0 .def_488)))
(let ((.def_524 (+ .def_518 .def_523)))
(let ((.def_473 (* b.delta__AT1 .def_472)))
(let ((.def_505 (* 2401.0 .def_473)))
(let ((.def_525 (+ .def_505 .def_524)))
(let ((.def_520 (* 25.0 b.speed_x__AT1)))
(let ((.def_526 (+ .def_520 .def_525)))
(let ((.def_545 (<= .def_526 0.0 )))
(let ((.def_546 (not .def_545)))
(let ((.def_593 (or .def_543 .def_546)))
(let ((.def_532 (not .def_531)))
(let ((.def_527 (<= 0.0 .def_526)))
(let ((.def_592 (or .def_527 .def_532)))
(let ((.def_594 (and .def_592 .def_593)))
(let ((.def_590 (or .def_587 .def_589)))
(let ((.def_591 (not .def_590)))
(let ((.def_595 (or .def_591 .def_594)))
(let ((.def_528 (not .def_527)))
(let ((.def_583 (or .def_528 .def_531)))
(let ((.def_544 (not .def_543)))
(let ((.def_582 (or .def_544 .def_545)))
(let ((.def_584 (and .def_582 .def_583)))
(let ((.def_580 (or .def_573 .def_579)))
(let ((.def_581 (not .def_580)))
(let ((.def_585 (or .def_581 .def_584)))
(let ((.def_596 (and .def_585 .def_595)))
(let ((.def_558 (and .def_543 .def_545)))
(let ((.def_557 (and .def_527 .def_531)))
(let ((.def_559 (or .def_557 .def_558)))
(let ((.def_489 (* b.delta__AT1 .def_488)))
(let ((.def_490 (* 200.0 .def_489)))
(let ((.def_485 (* .def_472 .def_484)))
(let ((.def_486 (* 200.0 .def_485)))
(let ((.def_496 (+ .def_486 .def_490)))
(let ((.def_481 (* .def_347 .def_472)))
(let ((.def_482 (* 100.0 .def_481)))
(let ((.def_497 (+ .def_482 .def_496)))
(let ((.def_478 (* .def_473 .def_477)))
(let ((.def_479 (* 200.0 .def_478)))
(let ((.def_498 (+ .def_479 .def_497)))
(let ((.def_474 (* b.delta__AT1 .def_473)))
(let ((.def_475 (* 2401.0 .def_474)))
(let ((.def_499 (+ .def_475 .def_498)))
(let ((.def_470 (* b.speed_x__AT1 b.delta__AT1)))
(let ((.def_471 (* 100.0 .def_470)))
(let ((.def_500 (+ .def_471 .def_499)))
(let ((.def_451 (* b.y__AT1 b.y__AT1)))
(let ((.def_492 (* 100.0 .def_451)))
(let ((.def_501 (+ .def_492 .def_500)))
(let ((.def_494 (* 100.0 b.x__AT1)))
(let ((.def_502 (+ .def_494 .def_501)))
(let ((.def_537 (<= .def_502 0.0 )))
(let ((.def_551 (not .def_537)))
(let ((.def_452 (+ b.x__AT1 .def_451)))
(let ((.def_535 (<= .def_452 0.0 )))
(let ((.def_552 (or .def_535 .def_551)))
(let ((.def_453 (<= 0.0 .def_452)))
(let ((.def_549 (not .def_453)))
(let ((.def_503 (<= 0.0 .def_502)))
(let ((.def_550 (or .def_503 .def_549)))
(let ((.def_553 (and .def_550 .def_552)))
(let ((.def_547 (or .def_544 .def_546)))
(let ((.def_548 (not .def_547)))
(let ((.def_554 (or .def_548 .def_553)))
(let ((.def_539 (not .def_503)))
(let ((.def_540 (or .def_453 .def_539)))
(let ((.def_536 (not .def_535)))
(let ((.def_538 (or .def_536 .def_537)))
(let ((.def_541 (and .def_538 .def_540)))
(let ((.def_533 (or .def_528 .def_532)))
(let ((.def_534 (not .def_533)))
(let ((.def_542 (or .def_534 .def_541)))
(let ((.def_555 (and .def_542 .def_554)))
(let ((.def_504 (and .def_453 .def_503)))
(let ((.def_556 (and .def_504 .def_555)))
(let ((.def_560 (and .def_556 .def_559)))
(let ((.def_597 (and .def_560 .def_596)))
(let ((.def_601 (and .def_597 .def_600)))
(let ((.def_627 (and .def_601 .def_626)))
(let ((.def_631 (and .def_627 .def_630)))
(let ((.def_635 (and .def_631 .def_634)))
(let ((.def_77 (<= b.speed_loss__AT0 (/ 1 2))))
(let ((.def_74 (<= (/ 1 10) b.speed_loss__AT0)))
(let ((.def_78 (and .def_74 .def_77)))
(let ((.def_636 (and .def_78 .def_635)))
(let ((.def_47 (not b.counter.0__AT0)))
(let ((.def_462 (or b.counter.1__AT0 .def_47)))
(let ((.def_463 (or b.counter.2__AT0 .def_462)))
(let ((.def_50 (not b.counter.2__AT0)))
(let ((.def_45 (not b.counter.1__AT0)))
(let ((.def_48 (and .def_45 .def_47)))
(let ((.def_461 (or .def_48 .def_50)))
(let ((.def_464 (and .def_461 .def_463)))
(let ((.def_465 (or b.counter.3__AT0 .def_464)))
(let ((.def_458 (or .def_48 b.counter.2__AT0)))
(let ((.def_457 (or .def_45 .def_50)))
(let ((.def_459 (and .def_457 .def_458)))
(let ((.def_53 (not b.counter.3__AT0)))
(let ((.def_460 (or .def_53 .def_459)))
(let ((.def_466 (and .def_460 .def_465)))
(let ((.def_454 (and .def_78 .def_453)))
(let ((.def_448 (not b.EVENT.0__AT1)))
(let ((.def_446 (not b.EVENT.1__AT1)))
(let ((.def_449 (or .def_446 .def_448)))
(let ((.def_9 (not b.counter.0__AT1)))
(let ((.def_6 (not b.counter.1__AT1)))
(let ((.def_439 (or .def_6 .def_9)))
(let ((.def_443 (or b.counter.3__AT1 .def_439)))
(let ((.def_440 (or b.counter.2__AT1 .def_439)))
(let ((.def_4 (not b.counter.2__AT1)))
(let ((.def_438 (or .def_4 .def_9)))
(let ((.def_441 (and .def_438 .def_440)))
(let ((.def_14 (not b.counter.3__AT1)))
(let ((.def_442 (or .def_14 .def_441)))
(let ((.def_444 (and .def_442 .def_443)))
(let ((.def_450 (and .def_444 .def_449)))
(let ((.def_455 (and .def_450 .def_454)))
(let ((.def_433 (<= 0.0 b.delta__AT0)))
(let ((.def_66 (not b.EVENT.0__AT0)))
(let ((.def_64 (not b.EVENT.1__AT0)))
(let ((.def_270 (and .def_64 .def_66)))
(let ((.def_272 (not .def_270)))
(let ((.def_434 (or .def_272 .def_433)))
(let ((.def_435 (or b.EVENT.1__AT0 .def_434)))
(let ((.def_312 (= b.bool_atom__AT0 b.bool_atom__AT1)))
(let ((.def_307 (= b.counter.0__AT1 b.counter.0__AT0)))
(let ((.def_305 (= b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_303 (= b.counter.2__AT1 b.counter.2__AT0)))
(let ((.def_302 (= b.counter.3__AT1 b.counter.3__AT0)))
(let ((.def_304 (and .def_302 .def_303)))
(let ((.def_306 (and .def_304 .def_305)))
(let ((.def_308 (and .def_306 .def_307)))
(let ((.def_429 (and .def_308 .def_312)))
(let ((.def_430 (or .def_272 .def_429)))
(let ((.def_431 (or b.EVENT.1__AT0 .def_430)))
(let ((.def_411 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_412 (* 10.0 .def_411)))
(let ((.def_416 (* (- 10.0) b.y__AT1)))
(let ((.def_420 (+ .def_416 .def_412)))
(let ((.def_87 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_413 (* (- 49.0) .def_87)))
(let ((.def_421 (+ .def_413 .def_420)))
(let ((.def_418 (* 10.0 b.y__AT0)))
(let ((.def_422 (+ .def_418 .def_421)))
(let ((.def_423 (= .def_422 0.0 )))
(let ((.def_406 (* (- 5.0) b.speed_y__AT1)))
(let ((.def_234 (* (- 49.0) b.delta__AT0)))
(let ((.def_407 (+ .def_234 .def_406)))
(let ((.def_235 (* 5.0 b.speed_y__AT0)))
(let ((.def_408 (+ .def_235 .def_407)))
(let ((.def_409 (= .def_408 0.0 )))
(let ((.def_402 (* (- 1.0) b.x__AT1)))
(let ((.def_84 (* b.speed_x__AT0 b.delta__AT0)))
(let ((.def_403 (+ .def_84 .def_402)))
(let ((.def_404 (+ b.x__AT0 .def_403)))
(let ((.def_405 (= .def_404 0.0 )))
(let ((.def_410 (and .def_405 .def_409)))
(let ((.def_424 (and .def_410 .def_423)))
(let ((.def_297 (= b.speed_x__AT0 b.speed_x__AT1)))
(let ((.def_425 (and .def_297 .def_424)))
(let ((.def_426 (or .def_272 .def_425)))
(let ((.def_294 (= b.y__AT0 b.y__AT1)))
(let ((.def_291 (= b.x__AT0 b.x__AT1)))
(let ((.def_396 (and .def_291 .def_294)))
(let ((.def_397 (and .def_297 .def_396)))
(let ((.def_300 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_398 (and .def_300 .def_397)))
(let ((.def_393 (= b.delta__AT0 0.0 )))
(let ((.def_394 (and .def_270 .def_393)))
(let ((.def_395 (not .def_394)))
(let ((.def_399 (or .def_395 .def_398)))
(let ((.def_400 (or b.EVENT.1__AT0 .def_399)))
(let ((.def_385 (and .def_297 .def_300)))
(let ((.def_386 (and .def_308 .def_385)))
(let ((.def_387 (or b.bool_atom__AT0 .def_386)))
(let ((.def_361 (or .def_9 b.counter.0__AT0)))
(let ((.def_360 (or b.counter.0__AT1 .def_47)))
(let ((.def_362 (and .def_360 .def_361)))
(let ((.def_363 (and .def_6 .def_362)))
(let ((.def_364 (or b.counter.1__AT0 .def_363)))
(let ((.def_359 (or b.counter.1__AT1 .def_45)))
(let ((.def_365 (and .def_359 .def_364)))
(let ((.def_376 (and .def_4 .def_365)))
(let ((.def_377 (or b.counter.2__AT0 .def_376)))
(let ((.def_371 (and .def_6 .def_47)))
(let ((.def_372 (or b.counter.1__AT0 .def_371)))
(let ((.def_373 (and .def_359 .def_372)))
(let ((.def_374 (and b.counter.2__AT1 .def_373)))
(let ((.def_375 (or .def_50 .def_374)))
(let ((.def_378 (and .def_375 .def_377)))
(let ((.def_379 (and b.counter.3__AT1 .def_378)))
(let ((.def_380 (or b.counter.3__AT0 .def_379)))
(let ((.def_366 (and b.counter.2__AT1 .def_365)))
(let ((.def_367 (or b.counter.2__AT0 .def_366)))
(let ((.def_355 (or b.counter.1__AT1 b.counter.1__AT0)))
(let ((.def_353 (and .def_6 b.counter.0__AT1)))
(let ((.def_354 (or .def_45 .def_353)))
(let ((.def_356 (and .def_354 .def_355)))
(let ((.def_357 (and .def_4 .def_356)))
(let ((.def_358 (or .def_50 .def_357)))
(let ((.def_368 (and .def_358 .def_367)))
(let ((.def_369 (and .def_14 .def_368)))
(let ((.def_370 (or .def_53 .def_369)))
(let ((.def_381 (and .def_370 .def_380)))
(let ((.def_342 (* b.speed_x__AT0 b.speed_x__AT0)))
(let ((.def_101 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_343 (+ .def_101 .def_342)))
(let ((.def_328 (* (- 1.0) b.speed_loss__AT0)))
(let ((.def_329 (+ 1.0 .def_328)))
(let ((.def_341 (* .def_329 .def_329)))
(let ((.def_344 (* .def_341 .def_343)))
(let ((.def_345 (* (- 1.0) .def_344)))
(let ((.def_349 (+ .def_345 .def_347)))
(let ((.def_340 (* b.speed_x__AT1 b.speed_x__AT1)))
(let ((.def_350 (+ .def_340 .def_349)))
(let ((.def_351 (= .def_350 0.0 )))
(let ((.def_318 (* 2.0 b.y__AT0)))
(let ((.def_332 (* .def_318 .def_329)))
(let ((.def_333 (* b.speed_y__AT0 .def_332)))
(let ((.def_330 (* b.speed_x__AT0 .def_329)))
(let ((.def_336 (+ .def_330 .def_333)))
(let ((.def_326 (* 2.0 b.y__AT1)))
(let ((.def_327 (* b.speed_y__AT1 .def_326)))
(let ((.def_337 (+ .def_327 .def_336)))
(let ((.def_338 (+ b.speed_x__AT1 .def_337)))
(let ((.def_339 (= .def_338 0.0 )))
(let ((.def_352 (and .def_339 .def_351)))
(let ((.def_382 (and .def_352 .def_381)))
(let ((.def_325 (not b.bool_atom__AT0)))
(let ((.def_383 (or .def_325 .def_382)))
(let ((.def_319 (* b.speed_y__AT0 .def_318)))
(let ((.def_320 (+ b.speed_x__AT0 .def_319)))
(let ((.def_321 (<= 0.0 .def_320)))
(let ((.def_322 (not .def_321)))
(let ((.def_69 (* b.y__AT0 b.y__AT0)))
(let ((.def_70 (+ b.x__AT0 .def_69)))
(let ((.def_317 (= .def_70 0.0 )))
(let ((.def_323 (and .def_317 .def_322)))
(let ((.def_324 (= b.bool_atom__AT0 .def_323)))
(let ((.def_384 (and .def_324 .def_383)))
(let ((.def_388 (and .def_384 .def_387)))
(let ((.def_389 (and .def_291 .def_388)))
(let ((.def_390 (and .def_294 .def_389)))
(let ((.def_391 (or .def_270 .def_390)))
(let ((.def_392 (or b.EVENT.1__AT0 .def_391)))
(let ((.def_401 (and .def_392 .def_400)))
(let ((.def_427 (and .def_401 .def_426)))
(let ((.def_277 (= b.time__AT0 b.time__AT1)))
(let ((.def_292 (and .def_277 .def_291)))
(let ((.def_295 (and .def_292 .def_294)))
(let ((.def_298 (and .def_295 .def_297)))
(let ((.def_301 (and .def_298 .def_300)))
(let ((.def_309 (and .def_301 .def_308)))
(let ((.def_313 (and .def_309 .def_312)))
(let ((.def_314 (or .def_64 .def_313)))
(let ((.def_281 (* (- 1.0) b.time__AT1)))
(let ((.def_284 (+ b.delta__AT0 .def_281)))
(let ((.def_285 (+ b.time__AT0 .def_284)))
(let ((.def_286 (= .def_285 0.0 )))
(let ((.def_287 (or .def_272 .def_286)))
(let ((.def_288 (or b.EVENT.1__AT0 .def_287)))
(let ((.def_278 (or .def_270 .def_277)))
(let ((.def_279 (or b.EVENT.1__AT0 .def_278)))
(let ((.def_289 (and .def_279 .def_288)))
(let ((.def_315 (and .def_289 .def_314)))
(let ((.def_274 (= .def_272 b.event_is_timed__AT1)))
(let ((.def_271 (= b.event_is_timed__AT0 .def_270)))
(let ((.def_275 (and .def_271 .def_274)))
(let ((.def_316 (and .def_275 .def_315)))
(let ((.def_428 (and .def_316 .def_427)))
(let ((.def_432 (and .def_428 .def_431)))
(let ((.def_436 (and .def_432 .def_435)))
(let ((.def_437 (and .def_64 .def_436)))
(let ((.def_456 (and .def_437 .def_455)))
(let ((.def_467 (and .def_456 .def_466)))
(let ((.def_237 (+ .def_235 .def_234)))
(let ((.def_250 (<= 0.0 .def_237)))
(let ((.def_251 (not .def_250)))
(let ((.def_248 (<= 0.0 b.speed_y__AT0)))
(let ((.def_265 (or .def_248 .def_251)))
(let ((.def_240 (<= b.speed_y__AT0 0.0 )))
(let ((.def_241 (not .def_240)))
(let ((.def_238 (<= .def_237 0.0 )))
(let ((.def_264 (or .def_238 .def_241)))
(let ((.def_266 (and .def_264 .def_265)))
(let ((.def_109 (* b.y__AT0 b.speed_y__AT0)))
(let ((.def_153 (* 2.0 .def_109)))
(let ((.def_154 (+ b.speed_x__AT0 .def_153)))
(let ((.def_167 (<= .def_154 0.0 )))
(let ((.def_261 (and .def_167 .def_250)))
(let ((.def_155 (<= 0.0 .def_154)))
(let ((.def_260 (and .def_155 .def_238)))
(let ((.def_262 (or .def_260 .def_261)))
(let ((.def_96 (* (- (/ 49 10)) b.speed_y__AT0)))
(let ((.def_128 (* 3.0 .def_96)))
(let ((.def_188 (* 2.0 .def_128)))
(let ((.def_189 (* b.delta__AT0 .def_188)))
(let ((.def_191 (* (- 50.0) .def_189)))
(let ((.def_193 (* (- 50.0) .def_101)))
(let ((.def_198 (+ .def_193 .def_191)))
(let ((.def_187 (* (- 7203.0) .def_87)))
(let ((.def_199 (+ .def_187 .def_198)))
(let ((.def_196 (* 490.0 b.y__AT0)))
(let ((.def_200 (+ .def_196 .def_199)))
(let ((.def_219 (<= 0.0 .def_200)))
(let ((.def_220 (not .def_219)))
(let ((.def_205 (* (- 5.0) .def_101)))
(let ((.def_206 (* 49.0 b.y__AT0)))
(let ((.def_208 (+ .def_206 .def_205)))
(let ((.def_217 (<= 0.0 .def_208)))
(let ((.def_255 (or .def_217 .def_220)))
(let ((.def_209 (<= .def_208 0.0 )))
(let ((.def_210 (not .def_209)))
(let ((.def_201 (<= .def_200 0.0 )))
(let ((.def_254 (or .def_201 .def_210)))
(let ((.def_256 (and .def_254 .def_255)))
(let ((.def_249 (not .def_248)))
(let ((.def_252 (or .def_249 .def_251)))
(let ((.def_253 (not .def_252)))
(let ((.def_257 (or .def_253 .def_256)))
(let ((.def_202 (not .def_201)))
(let ((.def_245 (or .def_202 .def_209)))
(let ((.def_218 (not .def_217)))
(let ((.def_244 (or .def_218 .def_219)))
(let ((.def_246 (and .def_244 .def_245)))
(let ((.def_239 (not .def_238)))
(let ((.def_242 (or .def_239 .def_241)))
(let ((.def_243 (not .def_242)))
(let ((.def_247 (or .def_243 .def_246)))
(let ((.def_258 (and .def_247 .def_257)))
(let ((.def_230 (and .def_167 .def_219)))
(let ((.def_229 (and .def_155 .def_201)))
(let ((.def_231 (or .def_229 .def_230)))
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
(let ((.def_169 (<= .def_150 0.0 )))
(let ((.def_170 (not .def_169)))
(let ((.def_224 (or .def_167 .def_170)))
(let ((.def_156 (not .def_155)))
(let ((.def_151 (<= 0.0 .def_150)))
(let ((.def_223 (or .def_151 .def_156)))
(let ((.def_225 (and .def_223 .def_224)))
(let ((.def_221 (or .def_218 .def_220)))
(let ((.def_222 (not .def_221)))
(let ((.def_226 (or .def_222 .def_225)))
(let ((.def_152 (not .def_151)))
(let ((.def_214 (or .def_152 .def_155)))
(let ((.def_168 (not .def_167)))
(let ((.def_213 (or .def_168 .def_169)))
(let ((.def_215 (and .def_213 .def_214)))
(let ((.def_211 (or .def_202 .def_210)))
(let ((.def_212 (not .def_211)))
(let ((.def_216 (or .def_212 .def_215)))
(let ((.def_227 (and .def_216 .def_226)))
(let ((.def_182 (and .def_167 .def_169)))
(let ((.def_181 (and .def_151 .def_155)))
(let ((.def_183 (or .def_181 .def_182)))
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
(let ((.def_161 (<= .def_123 0.0 )))
(let ((.def_175 (not .def_161)))
(let ((.def_159 (<= .def_70 0.0 )))
(let ((.def_176 (or .def_159 .def_175)))
(let ((.def_71 (<= 0.0 .def_70)))
(let ((.def_173 (not .def_71)))
(let ((.def_124 (<= 0.0 .def_123)))
(let ((.def_174 (or .def_124 .def_173)))
(let ((.def_177 (and .def_174 .def_176)))
(let ((.def_171 (or .def_168 .def_170)))
(let ((.def_172 (not .def_171)))
(let ((.def_178 (or .def_172 .def_177)))
(let ((.def_163 (not .def_124)))
(let ((.def_164 (or .def_71 .def_163)))
(let ((.def_160 (not .def_159)))
(let ((.def_162 (or .def_160 .def_161)))
(let ((.def_165 (and .def_162 .def_164)))
(let ((.def_157 (or .def_152 .def_156)))
(let ((.def_158 (not .def_157)))
(let ((.def_166 (or .def_158 .def_165)))
(let ((.def_179 (and .def_166 .def_178)))
(let ((.def_125 (and .def_71 .def_124)))
(let ((.def_180 (and .def_125 .def_179)))
(let ((.def_184 (and .def_180 .def_183)))
(let ((.def_228 (and .def_184 .def_227)))
(let ((.def_232 (and .def_228 .def_231)))
(let ((.def_259 (and .def_232 .def_258)))
(let ((.def_263 (and .def_259 .def_262)))
(let ((.def_267 (and .def_263 .def_266)))
(let ((.def_268 (and .def_78 .def_267)))
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
(let ((.def_269 (and .def_82 .def_268)))
(let ((.def_468 (and .def_269 .def_467)))
(let ((.def_637 (and .def_468 .def_636)))
.def_637))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
