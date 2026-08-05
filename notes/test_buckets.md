# Test file buckets by subsystem (Block 3.2)

From unique filenames in `notes/all_tests.txt` (collect-only).

| Bucket | Test files |
|--------|------------|
| safety/runtime | `test_certification_manifest_schema.py`, `test_per_platform_certification.py`, `test_product_run_certify.py`, `test_run_safety_gate_ci.py`, `test_runtime_mode.py`, `test_runtime_safety_gates.py`, `test_safety_gates_eval.py`, `test_safety_gates_yaml_load.py`, `test_stage_a_isolation.py`, `test_stage_a_key_bundle.py`, `test_stage_a_messages_and_fixtures.py`, `test_stage_a_types.py`, `test_therapy_safety.py`, `test_timing_enforcement.py`, `test_torch_stage_a_infer.py`, `test_torch_stage_a_runner_skeleton.py` |
| data | `test_advanced_augmentation.py`, `test_assistive_supervision.py`, `test_condition_specific.py`, `test_condition_tensor_contract.py`, `test_condition_tensor_forward.py`, `test_data_panoptic_and_video.py`, `test_dataset_pipeline.py`, `test_dataset_registry.py`, `test_frames_data_validation.py`, `test_gold_manifest.py`, `test_medallion_layout.py`, `test_video_clip_dataset.py`, `test_video_dataset_perf.py` |
| training | `test_gradnorm_integration.py`, `test_integration_constraints.py`, `test_loss_weighting.py`, `test_phase5_training.py`, `test_run_config_contract.py`, `test_sagemaker_config.py`, `test_sagemaker_integration.py`, `test_sagemaker_pipeline_entrypoint.py`, `test_temporal_supervision_loss.py`, `test_training_hardening.py`, `test_training_pipeline.py` |
| therapy | `test_production_rag_and_therapy_contracts.py`, `test_therapy.py`, `test_therapy_output_preferences_validation.py` |
| retrieval | `test_phase3_retrieval.py`, `test_rag_advisory.py`, `test_rag_reliability.py` |
| export | `test_export_temporal_smoke.py`, `test_export_validation.py` |
| other | `test_artifact_signing.py`, `test_comprehensive_system.py`, `test_connectivity_state_machine.py`, `test_critical_fixes.py`, `test_edge_cases.py`, `test_error_handling.py`, `test_haptic_urgency.py`, `test_hungarian_matcher_fixes.py`, `test_infra_validate_stubs.py`, `test_integration_structure.py`, `test_label_space_registry.py`, `test_local_rollback.py`, `test_ml_lifecycle.py`, `test_model.py`, `test_model_handle_resolution.py`, `test_model_release_iam_scope.py`, `test_multihead_benchmark.py`, `test_ops_launchers.py`, `test_optional_features_exist.py`, `test_ota_staging.py`, `test_performance.py`, `test_phase0_backbone.py`, `test_phase0_contracts.py`, `test_phase1_foundations.py`, `test_phase1_fusion.py`, `test_phase2_heads.py`, `test_phase4_knowledge.py`, `test_port_binding.py`, `test_production_hardening.py`, `test_production_remediation.py`, `test_scene_graph_consistency.py`, `test_signature_in_resolution.py`, `test_sprint_self_tests.py`, `test_stage_b_timeout.py`, `test_temporal_clip_targets.py`, `test_temporal_video_contract.py`, `test_video_manifest.py`, `test_video_panoptic_utils.py`, `test_video_preprocessing_pipeline.py` |

**Unique test files:** 87

Ambiguous / `other` note: names that do not clearly map (e.g. comprehensive_system, integration_constraints, optional_features, phase0) land in `other`.
