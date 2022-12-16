(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 4 and uses the quantifier free encoding with equivalences encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:42:25 2012
(declare-fun b.event_is_timed__AT3 () Bool)
(declare-fun b.y__AT0 () Real)
(declare-fun b.y__AT4 () Real)
(declare-fun b.y__AT3 () Real)
(declare-fun b.time__AT4 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.0__AT4 () Bool)
(declare-fun b.counter.0__AT3 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.delta__AT4 () Real)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.event_is_timed__AT1 () Bool)
(declare-fun b.counter.1__AT4 () Bool)
(declare-fun b.bool_atom__AT1 () Bool)
(declare-fun b.x__AT4 () Real)
(declare-fun b.counter.1__AT3 () Bool)
(declare-fun b.EVENT.0__AT4 () Bool)
(declare-fun b.counter.2__AT4 () Bool)
(declare-fun b.x__AT3 () Real)
(declare-fun b.counter.2__AT3 () Bool)
(declare-fun b.EVENT.1__AT4 () Bool)
(declare-fun b.counter.3__AT4 () Bool)
(declare-fun b.EVENT.1__AT3 () Bool)
(declare-fun b.counter.3__AT3 () Bool)
(declare-fun b.bool_atom__AT4 () Bool)
(declare-fun b.bool_atom__AT3 () Bool)
(declare-fun b.g__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.speed_y__AT4 () Real)
(declare-fun b.speed_y__AT3 () Real)
(declare-fun b.event_is_timed__AT2 () Bool)
(declare-fun b.x__AT1 () Real)
(declare-fun b.y__AT1 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.counter.0__AT1 () Bool)
(declare-fun b.counter.1__AT1 () Bool)
(declare-fun b.bool_atom__AT2 () Bool)
(declare-fun b.y__AT2 () Real)
(declare-fun b.counter.2__AT1 () Bool)
(declare-fun b.time__AT2 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.time__AT3 () Real)
(declare-fun b.counter.3__AT1 () Bool)
(declare-fun b.bool_atom__AT0 () Bool)
(declare-fun b.delta__AT2 () Real)
(declare-fun b.delta__AT3 () Real)
(declare-fun b.x__AT0 () Real)
(declare-fun b.x__AT2 () Real)
(declare-fun b.EVENT.0__AT2 () Bool)
(declare-fun b.EVENT.0__AT3 () Bool)
(declare-fun b.counter.0__AT2 () Bool)
(declare-fun b.EVENT.1__AT2 () Bool)
(declare-fun b.delta__AT1 () Real)
(declare-fun b.speed_y__AT1 () Real)
(declare-fun b.counter.1__AT2 () Bool)
(declare-fun b.time__AT1 () Real)
(declare-fun b.EVENT.0__AT1 () Bool)
(declare-fun b.counter.2__AT2 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.EVENT.1__AT1 () Bool)
(declare-fun b.counter.3__AT2 () Bool)
(declare-fun b.speed_y__AT2 () Real)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.event_is_timed__AT4 () Bool)
(assert (let ((.def_65 (* (- 1.0) b.g__AT0)))
(let ((.def_68 (* (/ 1 2) .def_65)))
(let ((.def_81 (* 2.0 .def_68)))
(let ((.def_807 (* .def_81 b.delta__AT4)))
(let ((.def_808 (+ b.speed_y__AT4 .def_807)))
(let ((.def_812 (<= .def_808 0.0 )))
(let ((.def_811 (<= b.speed_y__AT4 0.0 )))
(let ((.def_813 (and .def_811 .def_812)))
(let ((.def_809 (<= 0.0 .def_808)))
(let ((.def_806 (<= 0.0 b.speed_y__AT4)))
(let ((.def_810 (and .def_806 .def_809)))
(let ((.def_814 (or .def_810 .def_813)))
(let ((.def_797 (* b.speed_y__AT4 b.delta__AT4)))
(let ((.def_795 (* b.delta__AT4 b.delta__AT4)))
(let ((.def_796 (* .def_68 .def_795)))
(let ((.def_798 (+ .def_796 .def_797)))
(let ((.def_784 (* (- 1.0) b.x__AT4)))
(let ((.def_785 (* b.x__AT4 .def_784)))
(let ((.def_799 (* (- 1.0) .def_785)))
(let ((.def_802 (+ .def_799 .def_798)))
(let ((.def_803 (+ b.y__AT4 .def_802)))
(let ((.def_804 (<= 0.0 .def_803)))
(let ((.def_786 (<= .def_785 b.y__AT4)))
(let ((.def_805 (and .def_786 .def_804)))
(let ((.def_815 (and .def_805 .def_814)))
(let ((.def_57 (<= b.g__AT0 10.0 )))
(let ((.def_56 (<= 8.0 b.g__AT0)))
(let ((.def_58 (and .def_56 .def_57)))
(let ((.def_816 (and .def_58 .def_815)))
(let ((.def_524 (not b.counter.0__AT3)))
(let ((.def_514 (not b.counter.1__AT3)))
(let ((.def_790 (and .def_514 .def_524)))
(let ((.def_519 (not b.counter.2__AT3)))
(let ((.def_791 (and .def_519 .def_790)))
(let ((.def_787 (and .def_58 .def_786)))
(let ((.def_781 (not b.EVENT.0__AT4)))
(let ((.def_779 (not b.EVENT.1__AT4)))
(let ((.def_782 (or .def_779 .def_781)))
(let ((.def_6 (not b.counter.0__AT4)))
(let ((.def_4 (not b.counter.1__AT4)))
(let ((.def_772 (or .def_4 .def_6)))
(let ((.def_776 (or b.counter.3__AT4 .def_772)))
(let ((.def_773 (or b.counter.2__AT4 .def_772)))
(let ((.def_9 (not b.counter.2__AT4)))
(let ((.def_771 (or .def_6 .def_9)))
(let ((.def_774 (and .def_771 .def_773)))
(let ((.def_710 (not b.counter.3__AT4)))
(let ((.def_775 (or .def_710 .def_774)))
(let ((.def_777 (and .def_775 .def_776)))
(let ((.def_783 (and .def_777 .def_782)))
(let ((.def_788 (and .def_783 .def_787)))
(let ((.def_766 (<= 0.0 b.delta__AT3)))
(let ((.def_604 (not b.EVENT.0__AT3)))
(let ((.def_602 (not b.EVENT.1__AT3)))
(let ((.def_641 (and .def_602 .def_604)))
(let ((.def_643 (not .def_641)))
(let ((.def_767 (or .def_643 .def_766)))
(let ((.def_768 (or b.EVENT.1__AT3 .def_767)))
(let ((.def_679 (= b.bool_atom__AT3 b.bool_atom__AT4)))
(let ((.def_675 (= b.counter.0__AT4 b.counter.0__AT3)))
(let ((.def_673 (= b.counter.1__AT4 b.counter.1__AT3)))
(let ((.def_671 (= b.counter.2__AT4 b.counter.2__AT3)))
(let ((.def_670 (= b.counter.3__AT3 b.counter.3__AT4)))
(let ((.def_672 (and .def_670 .def_671)))
(let ((.def_674 (and .def_672 .def_673)))
(let ((.def_676 (and .def_674 .def_675)))
(let ((.def_762 (and .def_676 .def_679)))
(let ((.def_763 (or .def_643 .def_762)))
(let ((.def_764 (or b.EVENT.1__AT3 .def_763)))
(let ((.def_751 (* .def_65 b.delta__AT3)))
(let ((.def_752 (* (- 1.0) b.speed_y__AT4)))
(let ((.def_755 (+ .def_752 .def_751)))
(let ((.def_756 (+ b.speed_y__AT3 .def_755)))
(let ((.def_757 (= .def_756 0.0 )))
(let ((.def_742 (* (- 1.0) b.y__AT4)))
(let ((.def_620 (* b.speed_y__AT3 b.delta__AT3)))
(let ((.def_746 (+ .def_620 .def_742)))
(let ((.def_618 (* b.delta__AT3 b.delta__AT3)))
(let ((.def_619 (* .def_68 .def_618)))
(let ((.def_747 (+ .def_619 .def_746)))
(let ((.def_748 (+ b.y__AT3 .def_747)))
(let ((.def_749 (= .def_748 0.0 )))
(let ((.def_661 (= b.x__AT3 b.x__AT4)))
(let ((.def_750 (and .def_661 .def_749)))
(let ((.def_758 (and .def_750 .def_757)))
(let ((.def_759 (or .def_643 .def_758)))
(let ((.def_664 (= b.y__AT3 b.y__AT4)))
(let ((.def_737 (and .def_661 .def_664)))
(let ((.def_667 (= b.speed_y__AT3 b.speed_y__AT4)))
(let ((.def_738 (and .def_667 .def_737)))
(let ((.def_734 (= b.delta__AT3 0.0 )))
(let ((.def_735 (and .def_641 .def_734)))
(let ((.def_736 (not .def_735)))
(let ((.def_739 (or .def_736 .def_738)))
(let ((.def_740 (or b.EVENT.1__AT3 .def_739)))
(let ((.def_727 (and .def_667 .def_676)))
(let ((.def_728 (or b.bool_atom__AT3 .def_727)))
(let ((.def_702 (or .def_6 b.counter.0__AT3)))
(let ((.def_701 (or b.counter.0__AT4 .def_524)))
(let ((.def_703 (and .def_701 .def_702)))
(let ((.def_704 (and .def_4 .def_703)))
(let ((.def_705 (or b.counter.1__AT3 .def_704)))
(let ((.def_700 (or b.counter.1__AT4 .def_514)))
(let ((.def_706 (and .def_700 .def_705)))
(let ((.def_718 (and .def_9 .def_706)))
(let ((.def_719 (or b.counter.2__AT3 .def_718)))
(let ((.def_713 (and .def_4 .def_524)))
(let ((.def_714 (or b.counter.1__AT3 .def_713)))
(let ((.def_715 (and .def_700 .def_714)))
(let ((.def_716 (and b.counter.2__AT4 .def_715)))
(let ((.def_717 (or .def_519 .def_716)))
(let ((.def_720 (and .def_717 .def_719)))
(let ((.def_721 (and b.counter.3__AT4 .def_720)))
(let ((.def_722 (or b.counter.3__AT3 .def_721)))
(let ((.def_707 (and b.counter.2__AT4 .def_706)))
(let ((.def_708 (or b.counter.2__AT3 .def_707)))
(let ((.def_696 (or b.counter.1__AT4 b.counter.1__AT3)))
(let ((.def_694 (and .def_4 b.counter.0__AT4)))
(let ((.def_695 (or .def_514 .def_694)))
(let ((.def_697 (and .def_695 .def_696)))
(let ((.def_698 (and .def_9 .def_697)))
(let ((.def_699 (or .def_519 .def_698)))
(let ((.def_709 (and .def_699 .def_708)))
(let ((.def_711 (and .def_709 .def_710)))
(let ((.def_533 (not b.counter.3__AT3)))
(let ((.def_712 (or .def_533 .def_711)))
(let ((.def_723 (and .def_712 .def_722)))
(let ((.def_691 (* b.speed_y__AT4 b.speed_y__AT4)))
(let ((.def_511 (* b.speed_y__AT3 b.speed_y__AT3)))
(let ((.def_692 (= .def_511 .def_691)))
(let ((.def_689 (+ b.speed_y__AT3 b.speed_y__AT4)))
(let ((.def_690 (= .def_689 0.0 )))
(let ((.def_693 (and .def_690 .def_692)))
(let ((.def_724 (and .def_693 .def_723)))
(let ((.def_688 (not b.bool_atom__AT3)))
(let ((.def_725 (or .def_688 .def_724)))
(let ((.def_629 (<= 0.0 b.speed_y__AT3)))
(let ((.def_685 (not .def_629)))
(let ((.def_607 (* (- 1.0) b.x__AT3)))
(let ((.def_608 (* b.x__AT3 .def_607)))
(let ((.def_684 (= b.y__AT3 .def_608)))
(let ((.def_686 (and .def_684 .def_685)))
(let ((.def_687 (= b.bool_atom__AT3 .def_686)))
(let ((.def_726 (and .def_687 .def_725)))
(let ((.def_729 (and .def_726 .def_728)))
(let ((.def_730 (and .def_661 .def_729)))
(let ((.def_731 (and .def_664 .def_730)))
(let ((.def_732 (or .def_641 .def_731)))
(let ((.def_733 (or b.EVENT.1__AT3 .def_732)))
(let ((.def_741 (and .def_733 .def_740)))
(let ((.def_760 (and .def_741 .def_759)))
(let ((.def_648 (= b.time__AT3 b.time__AT4)))
(let ((.def_662 (and .def_648 .def_661)))
(let ((.def_665 (and .def_662 .def_664)))
(let ((.def_668 (and .def_665 .def_667)))
(let ((.def_677 (and .def_668 .def_676)))
(let ((.def_680 (and .def_677 .def_679)))
(let ((.def_681 (or .def_602 .def_680)))
(let ((.def_651 (* (- 1.0) b.time__AT4)))
(let ((.def_654 (+ b.delta__AT3 .def_651)))
(let ((.def_655 (+ b.time__AT3 .def_654)))
(let ((.def_656 (= .def_655 0.0 )))
(let ((.def_657 (or .def_643 .def_656)))
(let ((.def_658 (or b.EVENT.1__AT3 .def_657)))
(let ((.def_649 (or .def_641 .def_648)))
(let ((.def_650 (or b.EVENT.1__AT3 .def_649)))
(let ((.def_659 (and .def_650 .def_658)))
(let ((.def_682 (and .def_659 .def_681)))
(let ((.def_645 (= .def_643 b.event_is_timed__AT4)))
(let ((.def_642 (= b.event_is_timed__AT3 .def_641)))
(let ((.def_646 (and .def_642 .def_645)))
(let ((.def_683 (and .def_646 .def_682)))
(let ((.def_761 (and .def_683 .def_760)))
(let ((.def_765 (and .def_761 .def_764)))
(let ((.def_769 (and .def_765 .def_768)))
(let ((.def_770 (and .def_602 .def_769)))
(let ((.def_789 (and .def_770 .def_788)))
(let ((.def_792 (and .def_789 .def_791)))
(let ((.def_630 (* .def_81 b.delta__AT3)))
(let ((.def_631 (+ b.speed_y__AT3 .def_630)))
(let ((.def_635 (<= .def_631 0.0 )))
(let ((.def_634 (<= b.speed_y__AT3 0.0 )))
(let ((.def_636 (and .def_634 .def_635)))
(let ((.def_632 (<= 0.0 .def_631)))
(let ((.def_633 (and .def_629 .def_632)))
(let ((.def_637 (or .def_633 .def_636)))
(let ((.def_621 (+ .def_619 .def_620)))
(let ((.def_622 (* (- 1.0) .def_608)))
(let ((.def_625 (+ .def_622 .def_621)))
(let ((.def_626 (+ b.y__AT3 .def_625)))
(let ((.def_627 (<= 0.0 .def_626)))
(let ((.def_609 (<= .def_608 b.y__AT3)))
(let ((.def_628 (and .def_609 .def_627)))
(let ((.def_638 (and .def_628 .def_637)))
(let ((.def_639 (and .def_58 .def_638)))
(let ((.def_341 (not b.counter.0__AT2)))
(let ((.def_331 (not b.counter.1__AT2)))
(let ((.def_613 (and .def_331 .def_341)))
(let ((.def_336 (not b.counter.2__AT2)))
(let ((.def_614 (and .def_336 .def_613)))
(let ((.def_610 (and .def_58 .def_609)))
(let ((.def_605 (or .def_602 .def_604)))
(let ((.def_595 (or .def_514 .def_524)))
(let ((.def_599 (or b.counter.3__AT3 .def_595)))
(let ((.def_596 (or b.counter.2__AT3 .def_595)))
(let ((.def_594 (or .def_519 .def_524)))
(let ((.def_597 (and .def_594 .def_596)))
(let ((.def_598 (or .def_533 .def_597)))
(let ((.def_600 (and .def_598 .def_599)))
(let ((.def_606 (and .def_600 .def_605)))
(let ((.def_611 (and .def_606 .def_610)))
(let ((.def_589 (<= 0.0 b.delta__AT2)))
(let ((.def_421 (not b.EVENT.0__AT2)))
(let ((.def_419 (not b.EVENT.1__AT2)))
(let ((.def_458 (and .def_419 .def_421)))
(let ((.def_460 (not .def_458)))
(let ((.def_590 (or .def_460 .def_589)))
(let ((.def_591 (or b.EVENT.1__AT2 .def_590)))
(let ((.def_499 (= b.bool_atom__AT2 b.bool_atom__AT3)))
(let ((.def_495 (= b.counter.0__AT2 b.counter.0__AT3)))
(let ((.def_492 (= b.counter.1__AT2 b.counter.1__AT3)))
(let ((.def_489 (= b.counter.2__AT2 b.counter.2__AT3)))
(let ((.def_487 (= b.counter.3__AT2 b.counter.3__AT3)))
(let ((.def_490 (and .def_487 .def_489)))
(let ((.def_493 (and .def_490 .def_492)))
(let ((.def_496 (and .def_493 .def_495)))
(let ((.def_585 (and .def_496 .def_499)))
(let ((.def_586 (or .def_460 .def_585)))
(let ((.def_587 (or b.EVENT.1__AT2 .def_586)))
(let ((.def_574 (* .def_65 b.delta__AT2)))
(let ((.def_575 (* (- 1.0) b.speed_y__AT3)))
(let ((.def_578 (+ .def_575 .def_574)))
(let ((.def_579 (+ b.speed_y__AT2 .def_578)))
(let ((.def_580 (= .def_579 0.0 )))
(let ((.def_565 (* (- 1.0) b.y__AT3)))
(let ((.def_437 (* b.speed_y__AT2 b.delta__AT2)))
(let ((.def_569 (+ .def_437 .def_565)))
(let ((.def_435 (* b.delta__AT2 b.delta__AT2)))
(let ((.def_436 (* .def_68 .def_435)))
(let ((.def_570 (+ .def_436 .def_569)))
(let ((.def_571 (+ b.y__AT2 .def_570)))
(let ((.def_572 (= .def_571 0.0 )))
(let ((.def_478 (= b.x__AT2 b.x__AT3)))
(let ((.def_573 (and .def_478 .def_572)))
(let ((.def_581 (and .def_573 .def_580)))
(let ((.def_582 (or .def_460 .def_581)))
(let ((.def_481 (= b.y__AT2 b.y__AT3)))
(let ((.def_560 (and .def_478 .def_481)))
(let ((.def_484 (= b.speed_y__AT2 b.speed_y__AT3)))
(let ((.def_561 (and .def_484 .def_560)))
(let ((.def_557 (= b.delta__AT2 0.0 )))
(let ((.def_558 (and .def_458 .def_557)))
(let ((.def_559 (not .def_558)))
(let ((.def_562 (or .def_559 .def_561)))
(let ((.def_563 (or b.EVENT.1__AT2 .def_562)))
(let ((.def_550 (and .def_484 .def_496)))
(let ((.def_551 (or b.bool_atom__AT2 .def_550)))
(let ((.def_525 (or b.counter.0__AT2 .def_524)))
(let ((.def_523 (or .def_341 b.counter.0__AT3)))
(let ((.def_526 (and .def_523 .def_525)))
(let ((.def_527 (and .def_514 .def_526)))
(let ((.def_528 (or b.counter.1__AT2 .def_527)))
(let ((.def_522 (or .def_331 b.counter.1__AT3)))
(let ((.def_529 (and .def_522 .def_528)))
(let ((.def_541 (and .def_519 .def_529)))
(let ((.def_542 (or b.counter.2__AT2 .def_541)))
(let ((.def_536 (and .def_341 .def_514)))
(let ((.def_537 (or b.counter.1__AT2 .def_536)))
(let ((.def_538 (and .def_522 .def_537)))
(let ((.def_539 (and b.counter.2__AT3 .def_538)))
(let ((.def_540 (or .def_336 .def_539)))
(let ((.def_543 (and .def_540 .def_542)))
(let ((.def_544 (and b.counter.3__AT3 .def_543)))
(let ((.def_545 (or b.counter.3__AT2 .def_544)))
(let ((.def_530 (and b.counter.2__AT3 .def_529)))
(let ((.def_531 (or b.counter.2__AT2 .def_530)))
(let ((.def_517 (or b.counter.1__AT2 b.counter.1__AT3)))
(let ((.def_515 (and b.counter.0__AT3 .def_514)))
(let ((.def_516 (or .def_331 .def_515)))
(let ((.def_518 (and .def_516 .def_517)))
(let ((.def_520 (and .def_518 .def_519)))
(let ((.def_521 (or .def_336 .def_520)))
(let ((.def_532 (and .def_521 .def_531)))
(let ((.def_534 (and .def_532 .def_533)))
(let ((.def_350 (not b.counter.3__AT2)))
(let ((.def_535 (or .def_350 .def_534)))
(let ((.def_546 (and .def_535 .def_545)))
(let ((.def_328 (* b.speed_y__AT2 b.speed_y__AT2)))
(let ((.def_512 (= .def_328 .def_511)))
(let ((.def_509 (+ b.speed_y__AT2 b.speed_y__AT3)))
(let ((.def_510 (= .def_509 0.0 )))
(let ((.def_513 (and .def_510 .def_512)))
(let ((.def_547 (and .def_513 .def_546)))
(let ((.def_508 (not b.bool_atom__AT2)))
(let ((.def_548 (or .def_508 .def_547)))
(let ((.def_446 (<= 0.0 b.speed_y__AT2)))
(let ((.def_505 (not .def_446)))
(let ((.def_424 (* (- 1.0) b.x__AT2)))
(let ((.def_425 (* b.x__AT2 .def_424)))
(let ((.def_504 (= b.y__AT2 .def_425)))
(let ((.def_506 (and .def_504 .def_505)))
(let ((.def_507 (= b.bool_atom__AT2 .def_506)))
(let ((.def_549 (and .def_507 .def_548)))
(let ((.def_552 (and .def_549 .def_551)))
(let ((.def_553 (and .def_478 .def_552)))
(let ((.def_554 (and .def_481 .def_553)))
(let ((.def_555 (or .def_458 .def_554)))
(let ((.def_556 (or b.EVENT.1__AT2 .def_555)))
(let ((.def_564 (and .def_556 .def_563)))
(let ((.def_583 (and .def_564 .def_582)))
(let ((.def_465 (= b.time__AT2 b.time__AT3)))
(let ((.def_479 (and .def_465 .def_478)))
(let ((.def_482 (and .def_479 .def_481)))
(let ((.def_485 (and .def_482 .def_484)))
(let ((.def_497 (and .def_485 .def_496)))
(let ((.def_500 (and .def_497 .def_499)))
(let ((.def_501 (or .def_419 .def_500)))
(let ((.def_468 (* (- 1.0) b.time__AT3)))
(let ((.def_471 (+ b.delta__AT2 .def_468)))
(let ((.def_472 (+ b.time__AT2 .def_471)))
(let ((.def_473 (= .def_472 0.0 )))
(let ((.def_474 (or .def_460 .def_473)))
(let ((.def_475 (or b.EVENT.1__AT2 .def_474)))
(let ((.def_466 (or .def_458 .def_465)))
(let ((.def_467 (or b.EVENT.1__AT2 .def_466)))
(let ((.def_476 (and .def_467 .def_475)))
(let ((.def_502 (and .def_476 .def_501)))
(let ((.def_462 (= .def_460 b.event_is_timed__AT3)))
(let ((.def_459 (= b.event_is_timed__AT2 .def_458)))
(let ((.def_463 (and .def_459 .def_462)))
(let ((.def_503 (and .def_463 .def_502)))
(let ((.def_584 (and .def_503 .def_583)))
(let ((.def_588 (and .def_584 .def_587)))
(let ((.def_592 (and .def_588 .def_591)))
(let ((.def_593 (and .def_419 .def_592)))
(let ((.def_612 (and .def_593 .def_611)))
(let ((.def_615 (and .def_612 .def_614)))
(let ((.def_447 (* .def_81 b.delta__AT2)))
(let ((.def_448 (+ b.speed_y__AT2 .def_447)))
(let ((.def_452 (<= .def_448 0.0 )))
(let ((.def_451 (<= b.speed_y__AT2 0.0 )))
(let ((.def_453 (and .def_451 .def_452)))
(let ((.def_449 (<= 0.0 .def_448)))
(let ((.def_450 (and .def_446 .def_449)))
(let ((.def_454 (or .def_450 .def_453)))
(let ((.def_438 (+ .def_436 .def_437)))
(let ((.def_439 (* (- 1.0) .def_425)))
(let ((.def_442 (+ .def_439 .def_438)))
(let ((.def_443 (+ b.y__AT2 .def_442)))
(let ((.def_444 (<= 0.0 .def_443)))
(let ((.def_426 (<= .def_425 b.y__AT2)))
(let ((.def_445 (and .def_426 .def_444)))
(let ((.def_455 (and .def_445 .def_454)))
(let ((.def_456 (and .def_58 .def_455)))
(let ((.def_161 (not b.counter.0__AT1)))
(let ((.def_151 (not b.counter.1__AT1)))
(let ((.def_430 (and .def_151 .def_161)))
(let ((.def_156 (not b.counter.2__AT1)))
(let ((.def_431 (and .def_156 .def_430)))
(let ((.def_427 (and .def_58 .def_426)))
(let ((.def_422 (or .def_419 .def_421)))
(let ((.def_412 (or .def_331 .def_341)))
(let ((.def_416 (or b.counter.3__AT2 .def_412)))
(let ((.def_413 (or b.counter.2__AT2 .def_412)))
(let ((.def_411 (or .def_336 .def_341)))
(let ((.def_414 (and .def_411 .def_413)))
(let ((.def_415 (or .def_350 .def_414)))
(let ((.def_417 (and .def_415 .def_416)))
(let ((.def_423 (and .def_417 .def_422)))
(let ((.def_428 (and .def_423 .def_427)))
(let ((.def_406 (<= 0.0 b.delta__AT1)))
(let ((.def_240 (not b.EVENT.0__AT1)))
(let ((.def_238 (not b.EVENT.1__AT1)))
(let ((.def_275 (and .def_238 .def_240)))
(let ((.def_277 (not .def_275)))
(let ((.def_407 (or .def_277 .def_406)))
(let ((.def_408 (or b.EVENT.1__AT1 .def_407)))
(let ((.def_316 (= b.bool_atom__AT1 b.bool_atom__AT2)))
(let ((.def_312 (= b.counter.0__AT1 b.counter.0__AT2)))
(let ((.def_309 (= b.counter.1__AT1 b.counter.1__AT2)))
(let ((.def_306 (= b.counter.2__AT1 b.counter.2__AT2)))
(let ((.def_304 (= b.counter.3__AT1 b.counter.3__AT2)))
(let ((.def_307 (and .def_304 .def_306)))
(let ((.def_310 (and .def_307 .def_309)))
(let ((.def_313 (and .def_310 .def_312)))
(let ((.def_402 (and .def_313 .def_316)))
(let ((.def_403 (or .def_277 .def_402)))
(let ((.def_404 (or b.EVENT.1__AT1 .def_403)))
(let ((.def_391 (* .def_65 b.delta__AT1)))
(let ((.def_392 (* (- 1.0) b.speed_y__AT2)))
(let ((.def_395 (+ .def_392 .def_391)))
(let ((.def_396 (+ b.speed_y__AT1 .def_395)))
(let ((.def_397 (= .def_396 0.0 )))
(let ((.def_382 (* (- 1.0) b.y__AT2)))
(let ((.def_254 (* b.speed_y__AT1 b.delta__AT1)))
(let ((.def_386 (+ .def_254 .def_382)))
(let ((.def_252 (* b.delta__AT1 b.delta__AT1)))
(let ((.def_253 (* .def_68 .def_252)))
(let ((.def_387 (+ .def_253 .def_386)))
(let ((.def_388 (+ b.y__AT1 .def_387)))
(let ((.def_389 (= .def_388 0.0 )))
(let ((.def_295 (= b.x__AT1 b.x__AT2)))
(let ((.def_390 (and .def_295 .def_389)))
(let ((.def_398 (and .def_390 .def_397)))
(let ((.def_399 (or .def_277 .def_398)))
(let ((.def_298 (= b.y__AT1 b.y__AT2)))
(let ((.def_377 (and .def_295 .def_298)))
(let ((.def_301 (= b.speed_y__AT1 b.speed_y__AT2)))
(let ((.def_378 (and .def_301 .def_377)))
(let ((.def_374 (= b.delta__AT1 0.0 )))
(let ((.def_375 (and .def_275 .def_374)))
(let ((.def_376 (not .def_375)))
(let ((.def_379 (or .def_376 .def_378)))
(let ((.def_380 (or b.EVENT.1__AT1 .def_379)))
(let ((.def_367 (and .def_301 .def_313)))
(let ((.def_368 (or b.bool_atom__AT1 .def_367)))
(let ((.def_342 (or b.counter.0__AT1 .def_341)))
(let ((.def_340 (or .def_161 b.counter.0__AT2)))
(let ((.def_343 (and .def_340 .def_342)))
(let ((.def_344 (and .def_331 .def_343)))
(let ((.def_345 (or b.counter.1__AT1 .def_344)))
(let ((.def_339 (or .def_151 b.counter.1__AT2)))
(let ((.def_346 (and .def_339 .def_345)))
(let ((.def_358 (and .def_336 .def_346)))
(let ((.def_359 (or b.counter.2__AT1 .def_358)))
(let ((.def_353 (and .def_161 .def_331)))
(let ((.def_354 (or b.counter.1__AT1 .def_353)))
(let ((.def_355 (and .def_339 .def_354)))
(let ((.def_356 (and b.counter.2__AT2 .def_355)))
(let ((.def_357 (or .def_156 .def_356)))
(let ((.def_360 (and .def_357 .def_359)))
(let ((.def_361 (and b.counter.3__AT2 .def_360)))
(let ((.def_362 (or b.counter.3__AT1 .def_361)))
(let ((.def_347 (and b.counter.2__AT2 .def_346)))
(let ((.def_348 (or b.counter.2__AT1 .def_347)))
(let ((.def_334 (or b.counter.1__AT1 b.counter.1__AT2)))
(let ((.def_332 (and b.counter.0__AT2 .def_331)))
(let ((.def_333 (or .def_151 .def_332)))
(let ((.def_335 (and .def_333 .def_334)))
(let ((.def_337 (and .def_335 .def_336)))
(let ((.def_338 (or .def_156 .def_337)))
(let ((.def_349 (and .def_338 .def_348)))
(let ((.def_351 (and .def_349 .def_350)))
(let ((.def_170 (not b.counter.3__AT1)))
(let ((.def_352 (or .def_170 .def_351)))
(let ((.def_363 (and .def_352 .def_362)))
(let ((.def_147 (* b.speed_y__AT1 b.speed_y__AT1)))
(let ((.def_329 (= .def_147 .def_328)))
(let ((.def_326 (+ b.speed_y__AT1 b.speed_y__AT2)))
(let ((.def_327 (= .def_326 0.0 )))
(let ((.def_330 (and .def_327 .def_329)))
(let ((.def_364 (and .def_330 .def_363)))
(let ((.def_325 (not b.bool_atom__AT1)))
(let ((.def_365 (or .def_325 .def_364)))
(let ((.def_263 (<= 0.0 b.speed_y__AT1)))
(let ((.def_322 (not .def_263)))
(let ((.def_243 (* (- 1.0) b.x__AT1)))
(let ((.def_244 (* b.x__AT1 .def_243)))
(let ((.def_321 (= b.y__AT1 .def_244)))
(let ((.def_323 (and .def_321 .def_322)))
(let ((.def_324 (= b.bool_atom__AT1 .def_323)))
(let ((.def_366 (and .def_324 .def_365)))
(let ((.def_369 (and .def_366 .def_368)))
(let ((.def_370 (and .def_295 .def_369)))
(let ((.def_371 (and .def_298 .def_370)))
(let ((.def_372 (or .def_275 .def_371)))
(let ((.def_373 (or b.EVENT.1__AT1 .def_372)))
(let ((.def_381 (and .def_373 .def_380)))
(let ((.def_400 (and .def_381 .def_399)))
(let ((.def_282 (= b.time__AT1 b.time__AT2)))
(let ((.def_296 (and .def_282 .def_295)))
(let ((.def_299 (and .def_296 .def_298)))
(let ((.def_302 (and .def_299 .def_301)))
(let ((.def_314 (and .def_302 .def_313)))
(let ((.def_317 (and .def_314 .def_316)))
(let ((.def_318 (or .def_238 .def_317)))
(let ((.def_285 (* (- 1.0) b.time__AT2)))
(let ((.def_288 (+ b.delta__AT1 .def_285)))
(let ((.def_289 (+ b.time__AT1 .def_288)))
(let ((.def_290 (= .def_289 0.0 )))
(let ((.def_291 (or .def_277 .def_290)))
(let ((.def_292 (or b.EVENT.1__AT1 .def_291)))
(let ((.def_283 (or .def_275 .def_282)))
(let ((.def_284 (or b.EVENT.1__AT1 .def_283)))
(let ((.def_293 (and .def_284 .def_292)))
(let ((.def_319 (and .def_293 .def_318)))
(let ((.def_279 (= .def_277 b.event_is_timed__AT2)))
(let ((.def_276 (= b.event_is_timed__AT1 .def_275)))
(let ((.def_280 (and .def_276 .def_279)))
(let ((.def_320 (and .def_280 .def_319)))
(let ((.def_401 (and .def_320 .def_400)))
(let ((.def_405 (and .def_401 .def_404)))
(let ((.def_409 (and .def_405 .def_408)))
(let ((.def_410 (and .def_238 .def_409)))
(let ((.def_429 (and .def_410 .def_428)))
(let ((.def_432 (and .def_429 .def_431)))
(let ((.def_264 (* .def_81 b.delta__AT1)))
(let ((.def_265 (+ b.speed_y__AT1 .def_264)))
(let ((.def_269 (<= .def_265 0.0 )))
(let ((.def_268 (<= b.speed_y__AT1 0.0 )))
(let ((.def_270 (and .def_268 .def_269)))
(let ((.def_266 (<= 0.0 .def_265)))
(let ((.def_267 (and .def_263 .def_266)))
(let ((.def_271 (or .def_267 .def_270)))
(let ((.def_255 (+ .def_253 .def_254)))
(let ((.def_256 (* (- 1.0) .def_244)))
(let ((.def_259 (+ .def_256 .def_255)))
(let ((.def_260 (+ b.y__AT1 .def_259)))
(let ((.def_261 (<= 0.0 .def_260)))
(let ((.def_245 (<= .def_244 b.y__AT1)))
(let ((.def_262 (and .def_245 .def_261)))
(let ((.def_272 (and .def_262 .def_271)))
(let ((.def_273 (and .def_58 .def_272)))
(let ((.def_246 (and .def_58 .def_245)))
(let ((.def_241 (or .def_238 .def_240)))
(let ((.def_231 (or .def_151 .def_161)))
(let ((.def_235 (or b.counter.3__AT1 .def_231)))
(let ((.def_232 (or b.counter.2__AT1 .def_231)))
(let ((.def_230 (or .def_156 .def_161)))
(let ((.def_233 (and .def_230 .def_232)))
(let ((.def_234 (or .def_170 .def_233)))
(let ((.def_236 (and .def_234 .def_235)))
(let ((.def_242 (and .def_236 .def_241)))
(let ((.def_247 (and .def_242 .def_246)))
(let ((.def_225 (<= 0.0 b.delta__AT0)))
(let ((.def_46 (not b.EVENT.0__AT0)))
(let ((.def_44 (not b.EVENT.1__AT0)))
(let ((.def_93 (and .def_44 .def_46)))
(let ((.def_95 (not .def_93)))
(let ((.def_226 (or .def_95 .def_225)))
(let ((.def_227 (or b.EVENT.1__AT0 .def_226)))
(let ((.def_135 (= b.bool_atom__AT0 b.bool_atom__AT1)))
(let ((.def_130 (= b.counter.0__AT0 b.counter.0__AT1)))
(let ((.def_127 (= b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_124 (= b.counter.2__AT0 b.counter.2__AT1)))
(let ((.def_122 (= b.counter.3__AT0 b.counter.3__AT1)))
(let ((.def_125 (and .def_122 .def_124)))
(let ((.def_128 (and .def_125 .def_127)))
(let ((.def_131 (and .def_128 .def_130)))
(let ((.def_221 (and .def_131 .def_135)))
(let ((.def_222 (or .def_95 .def_221)))
(let ((.def_223 (or b.EVENT.1__AT0 .def_222)))
(let ((.def_210 (* b.delta__AT0 .def_65)))
(let ((.def_211 (* (- 1.0) b.speed_y__AT1)))
(let ((.def_214 (+ .def_211 .def_210)))
(let ((.def_215 (+ b.speed_y__AT0 .def_214)))
(let ((.def_216 (= .def_215 0.0 )))
(let ((.def_202 (* (- 1.0) b.y__AT1)))
(let ((.def_71 (* b.delta__AT0 b.speed_y__AT0)))
(let ((.def_205 (+ .def_71 .def_202)))
(let ((.def_64 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_69 (* .def_64 .def_68)))
(let ((.def_206 (+ .def_69 .def_205)))
(let ((.def_207 (+ b.y__AT0 .def_206)))
(let ((.def_208 (= .def_207 0.0 )))
(let ((.def_113 (= b.x__AT0 b.x__AT1)))
(let ((.def_209 (and .def_113 .def_208)))
(let ((.def_217 (and .def_209 .def_216)))
(let ((.def_218 (or .def_95 .def_217)))
(let ((.def_116 (= b.y__AT0 b.y__AT1)))
(let ((.def_197 (and .def_113 .def_116)))
(let ((.def_119 (= b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_198 (and .def_119 .def_197)))
(let ((.def_194 (= b.delta__AT0 0.0 )))
(let ((.def_195 (and .def_93 .def_194)))
(let ((.def_196 (not .def_195)))
(let ((.def_199 (or .def_196 .def_198)))
(let ((.def_200 (or b.EVENT.1__AT0 .def_199)))
(let ((.def_187 (and .def_119 .def_131)))
(let ((.def_188 (or b.bool_atom__AT0 .def_187)))
(let ((.def_162 (or b.counter.0__AT0 .def_161)))
(let ((.def_27 (not b.counter.0__AT0)))
(let ((.def_160 (or .def_27 b.counter.0__AT1)))
(let ((.def_163 (and .def_160 .def_162)))
(let ((.def_164 (and .def_151 .def_163)))
(let ((.def_165 (or b.counter.1__AT0 .def_164)))
(let ((.def_25 (not b.counter.1__AT0)))
(let ((.def_159 (or .def_25 b.counter.1__AT1)))
(let ((.def_166 (and .def_159 .def_165)))
(let ((.def_178 (and .def_156 .def_166)))
(let ((.def_179 (or b.counter.2__AT0 .def_178)))
(let ((.def_173 (and .def_27 .def_151)))
(let ((.def_174 (or b.counter.1__AT0 .def_173)))
(let ((.def_175 (and .def_159 .def_174)))
(let ((.def_176 (and b.counter.2__AT1 .def_175)))
(let ((.def_30 (not b.counter.2__AT0)))
(let ((.def_177 (or .def_30 .def_176)))
(let ((.def_180 (and .def_177 .def_179)))
(let ((.def_181 (and b.counter.3__AT1 .def_180)))
(let ((.def_182 (or b.counter.3__AT0 .def_181)))
(let ((.def_167 (and b.counter.2__AT1 .def_166)))
(let ((.def_168 (or b.counter.2__AT0 .def_167)))
(let ((.def_154 (or b.counter.1__AT0 b.counter.1__AT1)))
(let ((.def_152 (and b.counter.0__AT1 .def_151)))
(let ((.def_153 (or .def_25 .def_152)))
(let ((.def_155 (and .def_153 .def_154)))
(let ((.def_157 (and .def_155 .def_156)))
(let ((.def_158 (or .def_30 .def_157)))
(let ((.def_169 (and .def_158 .def_168)))
(let ((.def_171 (and .def_169 .def_170)))
(let ((.def_33 (not b.counter.3__AT0)))
(let ((.def_172 (or .def_33 .def_171)))
(let ((.def_183 (and .def_172 .def_182)))
(let ((.def_148 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_149 (= .def_147 .def_148)))
(let ((.def_145 (+ b.speed_y__AT0 b.speed_y__AT1)))
(let ((.def_146 (= .def_145 0.0 )))
(let ((.def_150 (and .def_146 .def_149)))
(let ((.def_184 (and .def_150 .def_183)))
(let ((.def_144 (not b.bool_atom__AT0)))
(let ((.def_185 (or .def_144 .def_184)))
(let ((.def_80 (<= 0.0 b.speed_y__AT0)))
(let ((.def_141 (not .def_80)))
(let ((.def_51 (* (- 1.0) b.x__AT0)))
(let ((.def_52 (* b.x__AT0 .def_51)))
(let ((.def_140 (= b.y__AT0 .def_52)))
(let ((.def_142 (and .def_140 .def_141)))
(let ((.def_143 (= b.bool_atom__AT0 .def_142)))
(let ((.def_186 (and .def_143 .def_185)))
(let ((.def_189 (and .def_186 .def_188)))
(let ((.def_190 (and .def_113 .def_189)))
(let ((.def_191 (and .def_116 .def_190)))
(let ((.def_192 (or .def_93 .def_191)))
(let ((.def_193 (or b.EVENT.1__AT0 .def_192)))
(let ((.def_201 (and .def_193 .def_200)))
(let ((.def_219 (and .def_201 .def_218)))
(let ((.def_100 (= b.time__AT0 b.time__AT1)))
(let ((.def_114 (and .def_100 .def_113)))
(let ((.def_117 (and .def_114 .def_116)))
(let ((.def_120 (and .def_117 .def_119)))
(let ((.def_132 (and .def_120 .def_131)))
(let ((.def_136 (and .def_132 .def_135)))
(let ((.def_137 (or .def_44 .def_136)))
(let ((.def_103 (* (- 1.0) b.time__AT1)))
(let ((.def_106 (+ b.delta__AT0 .def_103)))
(let ((.def_107 (+ b.time__AT0 .def_106)))
(let ((.def_108 (= .def_107 0.0 )))
(let ((.def_109 (or .def_95 .def_108)))
(let ((.def_110 (or b.EVENT.1__AT0 .def_109)))
(let ((.def_101 (or .def_93 .def_100)))
(let ((.def_102 (or b.EVENT.1__AT0 .def_101)))
(let ((.def_111 (and .def_102 .def_110)))
(let ((.def_138 (and .def_111 .def_137)))
(let ((.def_97 (= .def_95 b.event_is_timed__AT1)))
(let ((.def_94 (= b.event_is_timed__AT0 .def_93)))
(let ((.def_98 (and .def_94 .def_97)))
(let ((.def_139 (and .def_98 .def_138)))
(let ((.def_220 (and .def_139 .def_219)))
(let ((.def_224 (and .def_220 .def_223)))
(let ((.def_228 (and .def_224 .def_227)))
(let ((.def_229 (and .def_44 .def_228)))
(let ((.def_248 (and .def_229 .def_247)))
(let ((.def_28 (and .def_25 .def_27)))
(let ((.def_31 (and .def_28 .def_30)))
(let ((.def_249 (and .def_31 .def_248)))
(let ((.def_82 (* b.delta__AT0 .def_81)))
(let ((.def_83 (+ b.speed_y__AT0 .def_82)))
(let ((.def_87 (<= .def_83 0.0 )))
(let ((.def_86 (<= b.speed_y__AT0 0.0 )))
(let ((.def_88 (and .def_86 .def_87)))
(let ((.def_84 (<= 0.0 .def_83)))
(let ((.def_85 (and .def_80 .def_84)))
(let ((.def_89 (or .def_85 .def_88)))
(let ((.def_72 (+ .def_69 .def_71)))
(let ((.def_73 (* (- 1.0) .def_52)))
(let ((.def_76 (+ .def_73 .def_72)))
(let ((.def_77 (+ b.y__AT0 .def_76)))
(let ((.def_78 (<= 0.0 .def_77)))
(let ((.def_53 (<= .def_52 b.y__AT0)))
(let ((.def_79 (and .def_53 .def_78)))
(let ((.def_90 (and .def_79 .def_89)))
(let ((.def_91 (and .def_58 .def_90)))
(let ((.def_59 (and .def_53 .def_58)))
(let ((.def_47 (or .def_44 .def_46)))
(let ((.def_37 (or .def_25 .def_27)))
(let ((.def_41 (or b.counter.3__AT0 .def_37)))
(let ((.def_38 (or b.counter.2__AT0 .def_37)))
(let ((.def_36 (or .def_27 .def_30)))
(let ((.def_39 (and .def_36 .def_38)))
(let ((.def_40 (or .def_33 .def_39)))
(let ((.def_42 (and .def_40 .def_41)))
(let ((.def_48 (and .def_42 .def_47)))
(let ((.def_60 (and .def_48 .def_59)))
(let ((.def_34 (and .def_31 .def_33)))
(let ((.def_22 (= b.y__AT0 10.0 )))
(let ((.def_18 (= b.x__AT0 0.0 )))
(let ((.def_14 (= b.time__AT0 0.0 )))
(let ((.def_16 (and .def_14 b.event_is_timed__AT0)))
(let ((.def_19 (and .def_16 .def_18)))
(let ((.def_23 (and .def_19 .def_22)))
(let ((.def_35 (and .def_23 .def_34)))
(let ((.def_61 (and .def_35 .def_60)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_11 (not .def_10)))
(let ((.def_62 (and .def_11 .def_61)))
(let ((.def_92 (and .def_62 .def_91)))
(let ((.def_250 (and .def_92 .def_249)))
(let ((.def_274 (and .def_250 .def_273)))
(let ((.def_433 (and .def_274 .def_432)))
(let ((.def_457 (and .def_433 .def_456)))
(let ((.def_616 (and .def_457 .def_615)))
(let ((.def_640 (and .def_616 .def_639)))
(let ((.def_793 (and .def_640 .def_792)))
(let ((.def_817 (and .def_793 .def_816)))
.def_817))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)