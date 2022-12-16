(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the quantifier free encoding with equivalences encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:49:39 2012
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.EVENT.0__AT0 () Bool)
(assert (let ((.def_68 (* (- 49.0) b.delta__AT0)))
(let ((.def_70 (* 5.0 b.speed_y__AT0)))
(let ((.def_72 (+ .def_70 .def_68)))
(let ((.def_76 (<= .def_72 0.0 )))
(let ((.def_75 (<= b.speed_y__AT0 0.0 )))
(let ((.def_77 (and .def_75 .def_76)))
(let ((.def_73 (<= 0.0 .def_72)))
(let ((.def_67 (<= 0.0 b.speed_y__AT0)))
(let ((.def_74 (and .def_67 .def_73)))
(let ((.def_78 (or .def_74 .def_77)))
(let ((.def_59 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_60 (* 10.0 .def_59)))
(let ((.def_55 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_58 (* (- 49.0) .def_55)))
(let ((.def_61 (+ .def_58 .def_60)))
(let ((.def_62 (* 10.0 b.y__AT0)))
(let ((.def_64 (+ .def_62 .def_61)))
(let ((.def_65 (<= 0.0 .def_64)))
(let ((.def_41 (<= 0.0 b.y__AT0)))
(let ((.def_66 (and .def_41 .def_65)))
(let ((.def_79 (and .def_66 .def_78)))
(let ((.def_48 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_45 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_49 (and .def_45 .def_48)))
(let ((.def_80 (and .def_49 .def_79)))
(let ((.def_50 (and .def_41 .def_49)))
(let ((.def_38 (not b.EVENT.0__AT0)))
(let ((.def_36 (not b.EVENT.1__AT0)))
(let ((.def_39 (or .def_36 .def_38)))
(let ((.def_6 (not b.counter.0__AT0)))
(let ((.def_4 (not b.counter.1__AT0)))
(let ((.def_29 (or .def_4 .def_6)))
(let ((.def_33 (or b.counter.3__AT0 .def_29)))
(let ((.def_30 (or b.counter.2__AT0 .def_29)))
(let ((.def_9 (not b.counter.2__AT0)))
(let ((.def_28 (or .def_6 .def_9)))
(let ((.def_31 (and .def_28 .def_30)))
(let ((.def_25 (not b.counter.3__AT0)))
(let ((.def_32 (or .def_25 .def_31)))
(let ((.def_34 (and .def_32 .def_33)))
(let ((.def_40 (and .def_34 .def_39)))
(let ((.def_51 (and .def_40 .def_50)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_26 (and .def_10 .def_25)))
(let ((.def_22 (= b.speed_y__AT0 0.0 )))
(let ((.def_19 (= b.y__AT0 10.0 )))
(let ((.def_14 (= b.time__AT0 0.0 )))
(let ((.def_16 (and .def_14 b.event_is_timed__AT0)))
(let ((.def_20 (and .def_16 .def_19)))
(let ((.def_23 (and .def_20 .def_22)))
(let ((.def_27 (and .def_23 .def_26)))
(let ((.def_52 (and .def_27 .def_51)))
(let ((.def_11 (not .def_10)))
(let ((.def_53 (and .def_11 .def_52)))
(let ((.def_81 (and .def_53 .def_80)))
.def_81)))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
