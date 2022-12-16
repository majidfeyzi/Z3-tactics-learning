(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the quantifier free encoding with equivalences encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:44:02 2012
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
(assert (let ((.def_71 (* (- 49.0) b.delta__AT0)))
(let ((.def_73 (* 5.0 b.speed_y__AT0)))
(let ((.def_75 (+ .def_73 .def_71)))
(let ((.def_79 (<= .def_75 0.0 )))
(let ((.def_78 (<= b.speed_y__AT0 0.0 )))
(let ((.def_80 (and .def_78 .def_79)))
(let ((.def_76 (<= 0.0 .def_75)))
(let ((.def_70 (<= 0.0 b.speed_y__AT0)))
(let ((.def_77 (and .def_70 .def_76)))
(let ((.def_81 (or .def_77 .def_80)))
(let ((.def_62 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_63 (* 10.0 .def_62)))
(let ((.def_58 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_61 (* (- 49.0) .def_58)))
(let ((.def_64 (+ .def_61 .def_63)))
(let ((.def_65 (* 10.0 b.y__AT0)))
(let ((.def_67 (+ .def_65 .def_64)))
(let ((.def_68 (<= 0.0 .def_67)))
(let ((.def_44 (<= 0.0 b.y__AT0)))
(let ((.def_69 (and .def_44 .def_68)))
(let ((.def_82 (and .def_69 .def_81)))
(let ((.def_51 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_48 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_52 (and .def_48 .def_51)))
(let ((.def_83 (and .def_52 .def_82)))
(let ((.def_53 (and .def_44 .def_52)))
(let ((.def_41 (not b.EVENT.0__AT0)))
(let ((.def_39 (not b.EVENT.1__AT0)))
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
(let ((.def_54 (and .def_43 .def_53)))
(let ((.def_26 (and .def_4 .def_25)))
(let ((.def_27 (and .def_8 .def_26)))
(let ((.def_29 (and .def_27 .def_28)))
(let ((.def_23 (= b.speed_y__AT0 0.0 )))
(let ((.def_20 (= b.y__AT0 10.0 )))
(let ((.def_15 (= b.time__AT0 0.0 )))
(let ((.def_17 (and .def_15 b.event_is_timed__AT0)))
(let ((.def_21 (and .def_17 .def_20)))
(let ((.def_24 (and .def_21 .def_23)))
(let ((.def_30 (and .def_24 .def_29)))
(let ((.def_55 (and .def_30 .def_54)))
(let ((.def_6 (or .def_4 b.counter.1__AT0)))
(let ((.def_9 (or .def_6 .def_8)))
(let ((.def_11 (or .def_9 b.counter.3__AT0)))
(let ((.def_12 (not .def_11)))
(let ((.def_56 (and .def_12 .def_55)))
(let ((.def_84 (and .def_56 .def_83)))
.def_84))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
