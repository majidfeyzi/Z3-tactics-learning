(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 3 and uses the quantifier free encoding with lemmas encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:45:17 2012
(declare-fun b.delta__AT1 () Real)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.time__AT1 () Real)
(declare-fun b.speed_y__AT1 () Real)
(declare-fun b.speed_y__AT2 () Real)
(declare-fun b.time__AT0 () Real)
(declare-fun b.counter.2__AT2 () Bool)
(declare-fun b.counter.3__AT2 () Bool)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.event_is_timed__AT3 () Bool)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.event_is_timed__AT2 () Bool)
(declare-fun b.y__AT3 () Real)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.0__AT3 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.time__AT3 () Real)
(declare-fun b.counter.1__AT3 () Bool)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.0__AT2 () Bool)
(declare-fun b.delta__AT3 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.time__AT2 () Real)
(declare-fun b.counter.2__AT3 () Bool)
(declare-fun b.counter.1__AT2 () Bool)
(declare-fun b.EVENT.0__AT3 () Bool)
(declare-fun b.delta__AT2 () Real)
(declare-fun b.counter.3__AT3 () Bool)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.EVENT.1__AT3 () Bool)
(declare-fun b.EVENT.0__AT2 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.y__AT1 () Real)
(declare-fun b.y__AT2 () Real)
(declare-fun b.EVENT.1__AT2 () Bool)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.speed_y__AT3 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(assert (let ((.def_683 (<= b.speed_y__AT3 0.0 )))
(let ((.def_665 (* (- 49.0) b.delta__AT3)))
(let ((.def_666 (* 5.0 b.speed_y__AT3)))
(let ((.def_668 (+ .def_666 .def_665)))
(let ((.def_681 (<= .def_668 0.0 )))
(let ((.def_700 (and .def_681 .def_683)))
(let ((.def_669 (<= 0.0 .def_668)))
(let ((.def_670 (not .def_669)))
(let ((.def_663 (<= 0.0 b.speed_y__AT3)))
(let ((.def_697 (or .def_663 .def_670)))
(let ((.def_684 (not .def_683)))
(let ((.def_696 (or .def_681 .def_684)))
(let ((.def_698 (and .def_696 .def_697)))
(let ((.def_701 (and .def_698 .def_700)))
(let ((.def_695 (and .def_663 .def_669)))
(let ((.def_699 (and .def_695 .def_698)))
(let ((.def_702 (or .def_699 .def_701)))
(let ((.def_655 (* b.speed_y__AT3 b.delta__AT3)))
(let ((.def_656 (* 10.0 .def_655)))
(let ((.def_653 (* b.delta__AT3 b.delta__AT3)))
(let ((.def_654 (* (- 49.0) .def_653)))
(let ((.def_657 (+ .def_654 .def_656)))
(let ((.def_658 (* 10.0 b.y__AT3)))
(let ((.def_660 (+ .def_658 .def_657)))
(let ((.def_675 (<= .def_660 0.0 )))
(let ((.def_689 (not .def_675)))
(let ((.def_673 (<= b.y__AT3 0.0 )))
(let ((.def_690 (or .def_673 .def_689)))
(let ((.def_641 (<= 0.0 b.y__AT3)))
(let ((.def_687 (not .def_641)))
(let ((.def_661 (<= 0.0 .def_660)))
(let ((.def_688 (or .def_661 .def_687)))
(let ((.def_691 (and .def_688 .def_690)))
(let ((.def_682 (not .def_681)))
(let ((.def_685 (or .def_682 .def_684)))
(let ((.def_686 (not .def_685)))
(let ((.def_692 (or .def_686 .def_691)))
(let ((.def_677 (not .def_661)))
(let ((.def_678 (or .def_641 .def_677)))
(let ((.def_674 (not .def_673)))
(let ((.def_676 (or .def_674 .def_675)))
(let ((.def_679 (and .def_676 .def_678)))
(let ((.def_664 (not .def_663)))
(let ((.def_671 (or .def_664 .def_670)))
(let ((.def_672 (not .def_671)))
(let ((.def_680 (or .def_672 .def_679)))
(let ((.def_693 (and .def_680 .def_692)))
(let ((.def_662 (and .def_641 .def_661)))
(let ((.def_694 (and .def_662 .def_693)))
(let ((.def_703 (and .def_694 .def_702)))
(let ((.def_61 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_58 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_62 (and .def_58 .def_61)))
(let ((.def_704 (and .def_62 .def_703)))
(let ((.def_342 (not b.counter.0__AT2)))
(let ((.def_330 (not b.counter.1__AT2)))
(let ((.def_645 (and .def_330 .def_342)))
(let ((.def_648 (or b.counter.3__AT2 .def_645)))
(let ((.def_337 (not b.counter.2__AT2)))
(let ((.def_646 (and .def_337 .def_645)))
(let ((.def_352 (not b.counter.3__AT2)))
(let ((.def_647 (or .def_352 .def_646)))
(let ((.def_649 (and .def_647 .def_648)))
(let ((.def_642 (and .def_62 .def_641)))
(let ((.def_638 (not b.EVENT.0__AT3)))
(let ((.def_636 (not b.EVENT.1__AT3)))
(let ((.def_639 (or .def_636 .def_638)))
(let ((.def_6 (not b.counter.0__AT3)))
(let ((.def_4 (not b.counter.1__AT3)))
(let ((.def_629 (or .def_4 .def_6)))
(let ((.def_633 (or b.counter.3__AT3 .def_629)))
(let ((.def_630 (or b.counter.2__AT3 .def_629)))
(let ((.def_9 (not b.counter.2__AT3)))
(let ((.def_628 (or .def_6 .def_9)))
(let ((.def_631 (and .def_628 .def_630)))
(let ((.def_12 (not b.counter.3__AT3)))
(let ((.def_632 (or .def_12 .def_631)))
(let ((.def_634 (and .def_632 .def_633)))
(let ((.def_640 (and .def_634 .def_639)))
(let ((.def_643 (and .def_640 .def_642)))
(let ((.def_623 (<= 0.0 b.delta__AT2)))
(let ((.def_450 (not b.EVENT.0__AT2)))
(let ((.def_448 (not b.EVENT.1__AT2)))
(let ((.def_570 (and .def_448 .def_450)))
(let ((.def_574 (not .def_570)))
(let ((.def_624 (or .def_574 .def_623)))
(let ((.def_625 (or b.EVENT.1__AT2 .def_624)))
(let ((.def_562 (= b.counter.0__AT3 b.counter.0__AT2)))
(let ((.def_560 (= b.counter.1__AT3 b.counter.1__AT2)))
(let ((.def_558 (= b.counter.2__AT3 b.counter.2__AT2)))
(let ((.def_557 (= b.counter.3__AT3 b.counter.3__AT2)))
(let ((.def_559 (and .def_557 .def_558)))
(let ((.def_561 (and .def_559 .def_560)))
(let ((.def_563 (and .def_561 .def_562)))
(let ((.def_620 (or .def_563 .def_574)))
(let ((.def_621 (or b.EVENT.1__AT2 .def_620)))
(let ((.def_608 (* (- 10.0) b.y__AT3)))
(let ((.def_467 (* b.speed_y__AT2 b.delta__AT2)))
(let ((.def_468 (* 10.0 .def_467)))
(let ((.def_612 (+ .def_468 .def_608)))
(let ((.def_465 (* b.delta__AT2 b.delta__AT2)))
(let ((.def_466 (* (- 49.0) .def_465)))
(let ((.def_613 (+ .def_466 .def_612)))
(let ((.def_470 (* 10.0 b.y__AT2)))
(let ((.def_614 (+ .def_470 .def_613)))
(let ((.def_615 (= .def_614 0.0 )))
(let ((.def_604 (* (- 5.0) b.speed_y__AT3)))
(let ((.def_477 (* (- 49.0) b.delta__AT2)))
(let ((.def_605 (+ .def_477 .def_604)))
(let ((.def_478 (* 5.0 b.speed_y__AT2)))
(let ((.def_606 (+ .def_478 .def_605)))
(let ((.def_607 (= .def_606 0.0 )))
(let ((.def_616 (and .def_607 .def_615)))
(let ((.def_617 (or .def_574 .def_616)))
(let ((.def_568 (= b.y__AT2 b.y__AT3)))
(let ((.def_556 (= b.speed_y__AT2 b.speed_y__AT3)))
(let ((.def_601 (and .def_556 .def_568)))
(let ((.def_598 (= b.delta__AT2 0.0 )))
(let ((.def_599 (and .def_570 .def_598)))
(let ((.def_600 (not .def_599)))
(let ((.def_602 (or .def_600 .def_601)))
(let ((.def_603 (or b.EVENT.1__AT2 .def_602)))
(let ((.def_618 (and .def_603 .def_617)))
(let ((.def_580 (= b.time__AT2 b.time__AT3)))
(let ((.def_592 (and .def_568 .def_580)))
(let ((.def_593 (and .def_556 .def_592)))
(let ((.def_594 (and .def_563 .def_593)))
(let ((.def_595 (or .def_448 .def_594)))
(let ((.def_583 (* (- 1.0) b.time__AT3)))
(let ((.def_586 (+ b.delta__AT2 .def_583)))
(let ((.def_587 (+ b.time__AT2 .def_586)))
(let ((.def_588 (= .def_587 0.0 )))
(let ((.def_589 (or .def_574 .def_588)))
(let ((.def_590 (or b.EVENT.1__AT2 .def_589)))
(let ((.def_581 (or .def_570 .def_580)))
(let ((.def_582 (or b.EVENT.1__AT2 .def_581)))
(let ((.def_591 (and .def_582 .def_590)))
(let ((.def_596 (and .def_591 .def_595)))
(let ((.def_576 (= .def_574 b.event_is_timed__AT3)))
(let ((.def_573 (= b.event_is_timed__AT2 .def_570)))
(let ((.def_577 (and .def_573 .def_576)))
(let ((.def_564 (and .def_556 .def_563)))
(let ((.def_518 (= b.y__AT2 0.0 )))
(let ((.def_475 (<= 0.0 b.speed_y__AT2)))
(let ((.def_476 (not .def_475)))
(let ((.def_519 (and .def_476 .def_518)))
(let ((.def_565 (or .def_519 .def_564)))
(let ((.def_533 (or .def_6 b.counter.0__AT2)))
(let ((.def_532 (or b.counter.0__AT3 .def_342)))
(let ((.def_534 (and .def_532 .def_533)))
(let ((.def_535 (and .def_4 .def_534)))
(let ((.def_536 (or b.counter.1__AT2 .def_535)))
(let ((.def_531 (or b.counter.1__AT3 .def_330)))
(let ((.def_537 (and .def_531 .def_536)))
(let ((.def_548 (and .def_9 .def_537)))
(let ((.def_549 (or b.counter.2__AT2 .def_548)))
(let ((.def_543 (and .def_4 .def_342)))
(let ((.def_544 (or b.counter.1__AT2 .def_543)))
(let ((.def_545 (and .def_531 .def_544)))
(let ((.def_546 (and b.counter.2__AT3 .def_545)))
(let ((.def_547 (or .def_337 .def_546)))
(let ((.def_550 (and .def_547 .def_549)))
(let ((.def_551 (and b.counter.3__AT3 .def_550)))
(let ((.def_552 (or b.counter.3__AT2 .def_551)))
(let ((.def_538 (and b.counter.2__AT3 .def_537)))
(let ((.def_539 (or b.counter.2__AT2 .def_538)))
(let ((.def_527 (or b.counter.1__AT3 b.counter.1__AT2)))
(let ((.def_525 (and .def_4 b.counter.0__AT3)))
(let ((.def_526 (or .def_330 .def_525)))
(let ((.def_528 (and .def_526 .def_527)))
(let ((.def_529 (and .def_9 .def_528)))
(let ((.def_530 (or .def_337 .def_529)))
(let ((.def_540 (and .def_530 .def_539)))
(let ((.def_541 (and .def_12 .def_540)))
(let ((.def_542 (or .def_352 .def_541)))
(let ((.def_553 (and .def_542 .def_552)))
(let ((.def_521 (* (- 1.0) b.speed_y__AT2)))
(let ((.def_128 (* (- 1.0) speed_loss__AT0)))
(let ((.def_129 (+ 1.0 .def_128)))
(let ((.def_522 (* .def_129 .def_521)))
(let ((.def_524 (= .def_522 b.speed_y__AT3)))
(let ((.def_554 (and .def_524 .def_553)))
(let ((.def_520 (not .def_519)))
(let ((.def_555 (or .def_520 .def_554)))
(let ((.def_566 (and .def_555 .def_565)))
(let ((.def_569 (and .def_566 .def_568)))
(let ((.def_571 (or .def_569 .def_570)))
(let ((.def_572 (or b.EVENT.1__AT2 .def_571)))
(let ((.def_578 (and .def_572 .def_577)))
(let ((.def_597 (and .def_578 .def_596)))
(let ((.def_619 (and .def_597 .def_618)))
(let ((.def_622 (and .def_619 .def_621)))
(let ((.def_626 (and .def_622 .def_625)))
(let ((.def_627 (and .def_448 .def_626)))
(let ((.def_644 (and .def_627 .def_643)))
(let ((.def_650 (and .def_644 .def_649)))
(let ((.def_495 (<= b.speed_y__AT2 0.0 )))
(let ((.def_480 (+ .def_478 .def_477)))
(let ((.def_493 (<= .def_480 0.0 )))
(let ((.def_512 (and .def_493 .def_495)))
(let ((.def_481 (<= 0.0 .def_480)))
(let ((.def_482 (not .def_481)))
(let ((.def_509 (or .def_475 .def_482)))
(let ((.def_496 (not .def_495)))
(let ((.def_508 (or .def_493 .def_496)))
(let ((.def_510 (and .def_508 .def_509)))
(let ((.def_513 (and .def_510 .def_512)))
(let ((.def_507 (and .def_475 .def_481)))
(let ((.def_511 (and .def_507 .def_510)))
(let ((.def_514 (or .def_511 .def_513)))
(let ((.def_469 (+ .def_466 .def_468)))
(let ((.def_472 (+ .def_470 .def_469)))
(let ((.def_487 (<= .def_472 0.0 )))
(let ((.def_501 (not .def_487)))
(let ((.def_485 (<= b.y__AT2 0.0 )))
(let ((.def_502 (or .def_485 .def_501)))
(let ((.def_453 (<= 0.0 b.y__AT2)))
(let ((.def_499 (not .def_453)))
(let ((.def_473 (<= 0.0 .def_472)))
(let ((.def_500 (or .def_473 .def_499)))
(let ((.def_503 (and .def_500 .def_502)))
(let ((.def_494 (not .def_493)))
(let ((.def_497 (or .def_494 .def_496)))
(let ((.def_498 (not .def_497)))
(let ((.def_504 (or .def_498 .def_503)))
(let ((.def_489 (not .def_473)))
(let ((.def_490 (or .def_453 .def_489)))
(let ((.def_486 (not .def_485)))
(let ((.def_488 (or .def_486 .def_487)))
(let ((.def_491 (and .def_488 .def_490)))
(let ((.def_483 (or .def_476 .def_482)))
(let ((.def_484 (not .def_483)))
(let ((.def_492 (or .def_484 .def_491)))
(let ((.def_505 (and .def_492 .def_504)))
(let ((.def_474 (and .def_453 .def_473)))
(let ((.def_506 (and .def_474 .def_505)))
(let ((.def_515 (and .def_506 .def_514)))
(let ((.def_516 (and .def_62 .def_515)))
(let ((.def_147 (not b.counter.0__AT1)))
(let ((.def_135 (not b.counter.1__AT1)))
(let ((.def_457 (and .def_135 .def_147)))
(let ((.def_460 (or b.counter.3__AT1 .def_457)))
(let ((.def_142 (not b.counter.2__AT1)))
(let ((.def_458 (and .def_142 .def_457)))
(let ((.def_157 (not b.counter.3__AT1)))
(let ((.def_459 (or .def_157 .def_458)))
(let ((.def_461 (and .def_459 .def_460)))
(let ((.def_454 (and .def_62 .def_453)))
(let ((.def_451 (or .def_448 .def_450)))
(let ((.def_441 (or .def_330 .def_342)))
(let ((.def_445 (or b.counter.3__AT2 .def_441)))
(let ((.def_442 (or b.counter.2__AT2 .def_441)))
(let ((.def_440 (or .def_337 .def_342)))
(let ((.def_443 (and .def_440 .def_442)))
(let ((.def_444 (or .def_352 .def_443)))
(let ((.def_446 (and .def_444 .def_445)))
(let ((.def_452 (and .def_446 .def_451)))
(let ((.def_455 (and .def_452 .def_454)))
(let ((.def_435 (<= 0.0 b.delta__AT1)))
(let ((.def_256 (not b.EVENT.0__AT1)))
(let ((.def_254 (not b.EVENT.1__AT1)))
(let ((.def_382 (and .def_254 .def_256)))
(let ((.def_386 (not .def_382)))
(let ((.def_436 (or .def_386 .def_435)))
(let ((.def_437 (or b.EVENT.1__AT1 .def_436)))
(let ((.def_374 (= b.counter.0__AT1 b.counter.0__AT2)))
(let ((.def_372 (= b.counter.1__AT1 b.counter.1__AT2)))
(let ((.def_370 (= b.counter.2__AT1 b.counter.2__AT2)))
(let ((.def_369 (= b.counter.3__AT1 b.counter.3__AT2)))
(let ((.def_371 (and .def_369 .def_370)))
(let ((.def_373 (and .def_371 .def_372)))
(let ((.def_375 (and .def_373 .def_374)))
(let ((.def_432 (or .def_375 .def_386)))
(let ((.def_433 (or b.EVENT.1__AT1 .def_432)))
(let ((.def_420 (* (- 10.0) b.y__AT2)))
(let ((.def_271 (* b.speed_y__AT1 b.delta__AT1)))
(let ((.def_272 (* 10.0 .def_271)))
(let ((.def_424 (+ .def_272 .def_420)))
(let ((.def_269 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_270 (* (- 49.0) .def_269)))
(let ((.def_425 (+ .def_270 .def_424)))
(let ((.def_274 (* 10.0 b.y__AT1)))
(let ((.def_426 (+ .def_274 .def_425)))
(let ((.def_427 (= .def_426 0.0 )))
(let ((.def_416 (* (- 5.0) b.speed_y__AT2)))
(let ((.def_281 (* (- 49.0) b.delta__AT1)))
(let ((.def_417 (+ .def_281 .def_416)))
(let ((.def_282 (* 5.0 b.speed_y__AT1)))
(let ((.def_418 (+ .def_282 .def_417)))
(let ((.def_419 (= .def_418 0.0 )))
(let ((.def_428 (and .def_419 .def_427)))
(let ((.def_429 (or .def_386 .def_428)))
(let ((.def_380 (= b.y__AT1 b.y__AT2)))
(let ((.def_368 (= b.speed_y__AT1 b.speed_y__AT2)))
(let ((.def_413 (and .def_368 .def_380)))
(let ((.def_410 (= b.delta__AT1 0.0 )))
(let ((.def_411 (and .def_382 .def_410)))
(let ((.def_412 (not .def_411)))
(let ((.def_414 (or .def_412 .def_413)))
(let ((.def_415 (or b.EVENT.1__AT1 .def_414)))
(let ((.def_430 (and .def_415 .def_429)))
(let ((.def_392 (= b.time__AT1 b.time__AT2)))
(let ((.def_404 (and .def_380 .def_392)))
(let ((.def_405 (and .def_368 .def_404)))
(let ((.def_406 (and .def_375 .def_405)))
(let ((.def_407 (or .def_254 .def_406)))
(let ((.def_395 (* (- 1.0) b.time__AT2)))
(let ((.def_398 (+ b.delta__AT1 .def_395)))
(let ((.def_399 (+ b.time__AT1 .def_398)))
(let ((.def_400 (= .def_399 0.0 )))
(let ((.def_401 (or .def_386 .def_400)))
(let ((.def_402 (or b.EVENT.1__AT1 .def_401)))
(let ((.def_393 (or .def_382 .def_392)))
(let ((.def_394 (or b.EVENT.1__AT1 .def_393)))
(let ((.def_403 (and .def_394 .def_402)))
(let ((.def_408 (and .def_403 .def_407)))
(let ((.def_388 (= .def_386 b.event_is_timed__AT2)))
(let ((.def_385 (= b.event_is_timed__AT1 .def_382)))
(let ((.def_389 (and .def_385 .def_388)))
(let ((.def_376 (and .def_368 .def_375)))
(let ((.def_322 (= b.y__AT1 0.0 )))
(let ((.def_279 (<= 0.0 b.speed_y__AT1)))
(let ((.def_280 (not .def_279)))
(let ((.def_323 (and .def_280 .def_322)))
(let ((.def_377 (or .def_323 .def_376)))
(let ((.def_343 (or b.counter.0__AT1 .def_342)))
(let ((.def_341 (or .def_147 b.counter.0__AT2)))
(let ((.def_344 (and .def_341 .def_343)))
(let ((.def_345 (and .def_330 .def_344)))
(let ((.def_346 (or b.counter.1__AT1 .def_345)))
(let ((.def_340 (or .def_135 b.counter.1__AT2)))
(let ((.def_347 (and .def_340 .def_346)))
(let ((.def_360 (and .def_337 .def_347)))
(let ((.def_361 (or b.counter.2__AT1 .def_360)))
(let ((.def_355 (and .def_147 .def_330)))
(let ((.def_356 (or b.counter.1__AT1 .def_355)))
(let ((.def_357 (and .def_340 .def_356)))
(let ((.def_358 (and b.counter.2__AT2 .def_357)))
(let ((.def_359 (or .def_142 .def_358)))
(let ((.def_362 (and .def_359 .def_361)))
(let ((.def_363 (and b.counter.3__AT2 .def_362)))
(let ((.def_364 (or b.counter.3__AT1 .def_363)))
(let ((.def_348 (and b.counter.2__AT2 .def_347)))
(let ((.def_349 (or b.counter.2__AT1 .def_348)))
(let ((.def_334 (or b.counter.1__AT1 b.counter.1__AT2)))
(let ((.def_332 (and .def_330 b.counter.0__AT2)))
(let ((.def_333 (or .def_135 .def_332)))
(let ((.def_335 (and .def_333 .def_334)))
(let ((.def_338 (and .def_335 .def_337)))
(let ((.def_339 (or .def_142 .def_338)))
(let ((.def_350 (and .def_339 .def_349)))
(let ((.def_353 (and .def_350 .def_352)))
(let ((.def_354 (or .def_157 .def_353)))
(let ((.def_365 (and .def_354 .def_364)))
(let ((.def_325 (* (- 1.0) b.speed_y__AT1)))
(let ((.def_326 (* .def_129 .def_325)))
(let ((.def_328 (= .def_326 b.speed_y__AT2)))
(let ((.def_366 (and .def_328 .def_365)))
(let ((.def_324 (not .def_323)))
(let ((.def_367 (or .def_324 .def_366)))
(let ((.def_378 (and .def_367 .def_377)))
(let ((.def_381 (and .def_378 .def_380)))
(let ((.def_383 (or .def_381 .def_382)))
(let ((.def_384 (or b.EVENT.1__AT1 .def_383)))
(let ((.def_390 (and .def_384 .def_389)))
(let ((.def_409 (and .def_390 .def_408)))
(let ((.def_431 (and .def_409 .def_430)))
(let ((.def_434 (and .def_431 .def_433)))
(let ((.def_438 (and .def_434 .def_437)))
(let ((.def_439 (and .def_254 .def_438)))
(let ((.def_456 (and .def_439 .def_455)))
(let ((.def_462 (and .def_456 .def_461)))
(let ((.def_299 (<= b.speed_y__AT1 0.0 )))
(let ((.def_284 (+ .def_282 .def_281)))
(let ((.def_297 (<= .def_284 0.0 )))
(let ((.def_316 (and .def_297 .def_299)))
(let ((.def_285 (<= 0.0 .def_284)))
(let ((.def_286 (not .def_285)))
(let ((.def_313 (or .def_279 .def_286)))
(let ((.def_300 (not .def_299)))
(let ((.def_312 (or .def_297 .def_300)))
(let ((.def_314 (and .def_312 .def_313)))
(let ((.def_317 (and .def_314 .def_316)))
(let ((.def_311 (and .def_279 .def_285)))
(let ((.def_315 (and .def_311 .def_314)))
(let ((.def_318 (or .def_315 .def_317)))
(let ((.def_273 (+ .def_270 .def_272)))
(let ((.def_276 (+ .def_274 .def_273)))
(let ((.def_291 (<= .def_276 0.0 )))
(let ((.def_305 (not .def_291)))
(let ((.def_289 (<= b.y__AT1 0.0 )))
(let ((.def_306 (or .def_289 .def_305)))
(let ((.def_259 (<= 0.0 b.y__AT1)))
(let ((.def_303 (not .def_259)))
(let ((.def_277 (<= 0.0 .def_276)))
(let ((.def_304 (or .def_277 .def_303)))
(let ((.def_307 (and .def_304 .def_306)))
(let ((.def_298 (not .def_297)))
(let ((.def_301 (or .def_298 .def_300)))
(let ((.def_302 (not .def_301)))
(let ((.def_308 (or .def_302 .def_307)))
(let ((.def_293 (not .def_277)))
(let ((.def_294 (or .def_259 .def_293)))
(let ((.def_290 (not .def_289)))
(let ((.def_292 (or .def_290 .def_291)))
(let ((.def_295 (and .def_292 .def_294)))
(let ((.def_287 (or .def_280 .def_286)))
(let ((.def_288 (not .def_287)))
(let ((.def_296 (or .def_288 .def_295)))
(let ((.def_309 (and .def_296 .def_308)))
(let ((.def_278 (and .def_259 .def_277)))
(let ((.def_310 (and .def_278 .def_309)))
(let ((.def_319 (and .def_310 .def_318)))
(let ((.def_320 (and .def_62 .def_319)))
(let ((.def_32 (not b.counter.0__AT0)))
(let ((.def_30 (not b.counter.1__AT0)))
(let ((.def_33 (and .def_30 .def_32)))
(let ((.def_264 (or .def_33 b.counter.3__AT0)))
(let ((.def_38 (not b.counter.3__AT0)))
(let ((.def_35 (not b.counter.2__AT0)))
(let ((.def_36 (and .def_33 .def_35)))
(let ((.def_263 (or .def_36 .def_38)))
(let ((.def_265 (and .def_263 .def_264)))
(let ((.def_260 (and .def_62 .def_259)))
(let ((.def_257 (or .def_254 .def_256)))
(let ((.def_247 (or .def_135 .def_147)))
(let ((.def_251 (or b.counter.3__AT1 .def_247)))
(let ((.def_248 (or b.counter.2__AT1 .def_247)))
(let ((.def_246 (or .def_142 .def_147)))
(let ((.def_249 (and .def_246 .def_248)))
(let ((.def_250 (or .def_157 .def_249)))
(let ((.def_252 (and .def_250 .def_251)))
(let ((.def_258 (and .def_252 .def_257)))
(let ((.def_261 (and .def_258 .def_260)))
(let ((.def_241 (<= 0.0 b.delta__AT0)))
(let ((.def_51 (not b.EVENT.0__AT0)))
(let ((.def_49 (not b.EVENT.1__AT0)))
(let ((.def_187 (and .def_49 .def_51)))
(let ((.def_191 (not .def_187)))
(let ((.def_242 (or .def_191 .def_241)))
(let ((.def_243 (or b.EVENT.1__AT0 .def_242)))
(let ((.def_179 (= b.counter.0__AT0 b.counter.0__AT1)))
(let ((.def_177 (= b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_175 (= b.counter.2__AT0 b.counter.2__AT1)))
(let ((.def_174 (= b.counter.3__AT0 b.counter.3__AT1)))
(let ((.def_176 (and .def_174 .def_175)))
(let ((.def_178 (and .def_176 .def_177)))
(let ((.def_180 (and .def_178 .def_179)))
(let ((.def_238 (or .def_180 .def_191)))
(let ((.def_239 (or b.EVENT.1__AT0 .def_238)))
(let ((.def_227 (* (- 10.0) b.y__AT1)))
(let ((.def_72 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_73 (* 10.0 .def_72)))
(let ((.def_230 (+ .def_73 .def_227)))
(let ((.def_68 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_71 (* (- 49.0) .def_68)))
(let ((.def_231 (+ .def_71 .def_230)))
(let ((.def_75 (* 10.0 b.y__AT0)))
(let ((.def_232 (+ .def_75 .def_231)))
(let ((.def_233 (= .def_232 0.0 )))
(let ((.def_222 (* (- 5.0) b.speed_y__AT1)))
(let ((.def_82 (* (- 49.0) b.delta__AT0)))
(let ((.def_223 (+ .def_82 .def_222)))
(let ((.def_84 (* 5.0 b.speed_y__AT0)))
(let ((.def_224 (+ .def_84 .def_223)))
(let ((.def_225 (= .def_224 0.0 )))
(let ((.def_234 (and .def_225 .def_233)))
(let ((.def_235 (or .def_191 .def_234)))
(let ((.def_185 (= b.y__AT0 b.y__AT1)))
(let ((.def_173 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_218 (and .def_173 .def_185)))
(let ((.def_215 (= b.delta__AT0 0.0 )))
(let ((.def_216 (and .def_187 .def_215)))
(let ((.def_217 (not .def_216)))
(let ((.def_219 (or .def_217 .def_218)))
(let ((.def_220 (or b.EVENT.1__AT0 .def_219)))
(let ((.def_236 (and .def_220 .def_235)))
(let ((.def_197 (= b.time__AT0 b.time__AT1)))
(let ((.def_209 (and .def_185 .def_197)))
(let ((.def_210 (and .def_173 .def_209)))
(let ((.def_211 (and .def_180 .def_210)))
(let ((.def_212 (or .def_49 .def_211)))
(let ((.def_200 (* (- 1.0) b.time__AT1)))
(let ((.def_203 (+ b.delta__AT0 .def_200)))
(let ((.def_204 (+ b.time__AT0 .def_203)))
(let ((.def_205 (= .def_204 0.0 )))
(let ((.def_206 (or .def_191 .def_205)))
(let ((.def_207 (or b.EVENT.1__AT0 .def_206)))
(let ((.def_198 (or .def_187 .def_197)))
(let ((.def_199 (or b.EVENT.1__AT0 .def_198)))
(let ((.def_208 (and .def_199 .def_207)))
(let ((.def_213 (and .def_208 .def_212)))
(let ((.def_193 (= .def_191 b.event_is_timed__AT1)))
(let ((.def_190 (= b.event_is_timed__AT0 .def_187)))
(let ((.def_194 (and .def_190 .def_193)))
(let ((.def_181 (and .def_173 .def_180)))
(let ((.def_124 (= b.y__AT0 0.0 )))
(let ((.def_80 (<= 0.0 b.speed_y__AT0)))
(let ((.def_81 (not .def_80)))
(let ((.def_125 (and .def_81 .def_124)))
(let ((.def_182 (or .def_125 .def_181)))
(let ((.def_148 (or b.counter.0__AT0 .def_147)))
(let ((.def_146 (or .def_32 b.counter.0__AT1)))
(let ((.def_149 (and .def_146 .def_148)))
(let ((.def_150 (and .def_135 .def_149)))
(let ((.def_151 (or b.counter.1__AT0 .def_150)))
(let ((.def_145 (or .def_30 b.counter.1__AT1)))
(let ((.def_152 (and .def_145 .def_151)))
(let ((.def_165 (and .def_142 .def_152)))
(let ((.def_166 (or b.counter.2__AT0 .def_165)))
(let ((.def_160 (and .def_32 .def_135)))
(let ((.def_161 (or b.counter.1__AT0 .def_160)))
(let ((.def_162 (and .def_145 .def_161)))
(let ((.def_163 (and b.counter.2__AT1 .def_162)))
(let ((.def_164 (or .def_35 .def_163)))
(let ((.def_167 (and .def_164 .def_166)))
(let ((.def_168 (and b.counter.3__AT1 .def_167)))
(let ((.def_169 (or b.counter.3__AT0 .def_168)))
(let ((.def_153 (and b.counter.2__AT1 .def_152)))
(let ((.def_154 (or b.counter.2__AT0 .def_153)))
(let ((.def_139 (or b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_137 (and .def_135 b.counter.0__AT1)))
(let ((.def_138 (or .def_30 .def_137)))
(let ((.def_140 (and .def_138 .def_139)))
(let ((.def_143 (and .def_140 .def_142)))
(let ((.def_144 (or .def_35 .def_143)))
(let ((.def_155 (and .def_144 .def_154)))
(let ((.def_158 (and .def_155 .def_157)))
(let ((.def_159 (or .def_38 .def_158)))
(let ((.def_170 (and .def_159 .def_169)))
(let ((.def_130 (* (- 1.0) b.speed_y__AT0)))
(let ((.def_131 (* .def_129 .def_130)))
(let ((.def_133 (= .def_131 b.speed_y__AT1)))
(let ((.def_171 (and .def_133 .def_170)))
(let ((.def_126 (not .def_125)))
(let ((.def_172 (or .def_126 .def_171)))
(let ((.def_183 (and .def_172 .def_182)))
(let ((.def_186 (and .def_183 .def_185)))
(let ((.def_188 (or .def_186 .def_187)))
(let ((.def_189 (or b.EVENT.1__AT0 .def_188)))
(let ((.def_195 (and .def_189 .def_194)))
(let ((.def_214 (and .def_195 .def_213)))
(let ((.def_237 (and .def_214 .def_236)))
(let ((.def_240 (and .def_237 .def_239)))
(let ((.def_244 (and .def_240 .def_243)))
(let ((.def_245 (and .def_49 .def_244)))
(let ((.def_262 (and .def_245 .def_261)))
(let ((.def_266 (and .def_262 .def_265)))
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
(let ((.def_14 (or .def_7 b.counter.3__AT3)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_13 (or .def_10 .def_12)))
(let ((.def_15 (and .def_13 .def_14)))
(let ((.def_16 (not .def_15)))
(let ((.def_66 (and .def_16 .def_65)))
(let ((.def_123 (and .def_66 .def_122)))
(let ((.def_267 (and .def_123 .def_266)))
(let ((.def_321 (and .def_267 .def_320)))
(let ((.def_463 (and .def_321 .def_462)))
(let ((.def_517 (and .def_463 .def_516)))
(let ((.def_651 (and .def_517 .def_650)))
(let ((.def_705 (and .def_651 .def_704)))
.def_705)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
