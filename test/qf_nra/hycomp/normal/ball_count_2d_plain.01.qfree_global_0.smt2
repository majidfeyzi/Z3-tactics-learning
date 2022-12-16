(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the encoding obtained with quantifier elimination using qepcad encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:46:41 2012
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun g__AT0 () Real)
(declare-fun b.counter.2__AT0 () Bool)
(declare-fun b.delta__AT0 () Real)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun b.y__AT1 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun speed_loss__AT0 () Real)
(assert (let ((.def_98 (= b.y__AT0 b.y__AT1)))
(let ((.def_99 (not .def_98)))
(let ((.def_71 (* 2.0 b.delta__AT0)))
(let ((.def_72 (* b.speed_y__AT0 .def_71)))
(let ((.def_67 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_68 (* g__AT0 .def_67)))
(let ((.def_70 (* (- 1.0) .def_68)))
(let ((.def_73 (+ .def_70 .def_72)))
(let ((.def_74 (* 2.0 b.y__AT0)))
(let ((.def_76 (+ .def_74 .def_73)))
(let ((.def_91 (<= .def_76 0.0 )))
(let ((.def_92 (not .def_91)))
(let ((.def_88 (* b.speed_y__AT0 b.speed_y__AT0)))
(let ((.def_87 (* g__AT0 .def_74)))
(let ((.def_89 (+ .def_87 .def_88)))
(let ((.def_90 (<= .def_89 0.0 )))
(let ((.def_93 (and .def_90 .def_92)))
(let ((.def_38 (not b.EVENT.0__AT0)))
(let ((.def_36 (not b.EVENT.1__AT0)))
(let ((.def_60 (and .def_36 .def_38)))
(let ((.def_94 (and .def_60 .def_93)))
(let ((.def_81 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_82 (+ .def_74 .def_81)))
(let ((.def_83 (<= 0.0 .def_82)))
(let ((.def_65 (<= 0.0 b.speed_y__AT0)))
(let ((.def_80 (not .def_65)))
(let ((.def_84 (and .def_80 .def_83)))
(let ((.def_77 (<= 0.0 .def_76)))
(let ((.def_85 (and .def_77 .def_84)))
(let ((.def_55 (<= 0.0 b.y__AT0)))
(let ((.def_66 (and .def_55 .def_65)))
(let ((.def_78 (and .def_66 .def_77)))
(let ((.def_63 (<= 0.0 b.delta__AT0)))
(let ((.def_64 (not .def_63)))
(let ((.def_79 (or .def_64 .def_78)))
(let ((.def_86 (or .def_79 .def_85)))
(let ((.def_95 (or .def_86 .def_94)))
(let ((.def_96 (not .def_95)))
(let ((.def_100 (or .def_96 .def_99)))
(let ((.def_61 (not .def_60)))
(let ((.def_101 (or .def_61 .def_100)))
(let ((.def_52 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_49 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_53 (and .def_49 .def_52)))
(let ((.def_44 (<= g__AT0 10.0 )))
(let ((.def_43 (<= 8.0 g__AT0)))
(let ((.def_45 (and .def_43 .def_44)))
(let ((.def_54 (and .def_45 .def_53)))
(let ((.def_56 (and .def_54 .def_55)))
(let ((.def_39 (or .def_36 .def_38)))
(let ((.def_6 (not b.counter.0__AT0)))
(let ((.def_4 (not b.counter.1__AT0)))
(let ((.def_29 (or .def_4 .def_6)))
(let ((.def_33 (or b.counter.3__AT0 .def_29)))
(let ((.def_30 (or b.counter.2__AT0 .def_29)))
(let ((.def_9 (not b.counter.2__AT0)))
(let ((.def_28 (or .def_6 .def_9)))
(let ((.def_31 (and .def_28 .def_30)))
(let ((.def_12 (not b.counter.3__AT0)))
(let ((.def_32 (or .def_12 .def_31)))
(let ((.def_34 (and .def_32 .def_33)))
(let ((.def_40 (and .def_34 .def_39)))
(let ((.def_57 (and .def_40 .def_56)))
(let ((.def_25 (= b.speed_y__AT0 0.0 )))
(let ((.def_22 (= b.y__AT0 10.0 )))
(let ((.def_17 (= b.time__AT0 0.0 )))
(let ((.def_19 (and .def_17 b.event_is_timed__AT0)))
(let ((.def_23 (and .def_19 .def_22)))
(let ((.def_26 (and .def_23 .def_25)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_13 (and .def_10 .def_12)))
(let ((.def_27 (and .def_13 .def_26)))
(let ((.def_58 (and .def_27 .def_57)))
(let ((.def_14 (not .def_13)))
(let ((.def_59 (and .def_14 .def_58)))
(let ((.def_102 (and .def_59 .def_101)))
.def_102))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
