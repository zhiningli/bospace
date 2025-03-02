from src.database import ResultRepository
from src.database import ScriptRepository


results = ResultRepository.get_all_results()


dict_to_evaluate = {}
for result_object in results:
    model_idx = result_object.model_idx
    dataset_idx = result_object.dataset_idx
    if model_idx > 15 or dataset_idx > 14:
        continue
    script_idx = result_object.script_idx


    if script_idx not in dict_to_evaluate.keys():
        dict_to_evaluate[script_idx] = {}
    result_type = result_object.result_type
    dict_to_evaluate[script_idx][result_type] = result_object.sgd_best_performing_configuration["highest_score"]

test_dict = {}
for result_object in results:
    model_idx = result_object.model_idx
    dataset_idx = result_object.dataset_idx
    script_idx = result_object.script_idx
    if model_idx > 15 or dataset_idx > 14:
        continue
    if script_idx not in test_dict.keys():
        test_dict[script_idx] = {"highest_score": 0}
    if result_object.sgd_best_performing_configuration["highest_score"] >= test_dict[script_idx]["highest_score"]:
        test_dict[script_idx] = result_object.sgd_best_performing_configuration


print(len(test_dict.keys()))

for script_idx in test_dict:
    script = ScriptRepository.get_script_by_script_idx(script_idx=script_idx)

    script_performance = script.sgd_best_performing_configuration
    if script_idx == "2":
        print(script_performance)
    if ( script_performance["highest_score"] < test_dict[script_idx]["highest_score"]
        ):
        print(script_performance)
        print(test_dict[script_idx])

        print("Oh no, data is corrupted")

# script_performance["learning_rate"] != test_dict[script_idx]["learning_rate"] or
#         script_performance["weight_decay"] != test_dict[script_idx]["weight_decay"] or 
#         script_performance["momentum"] != test_dict[script_idx]["momentum"] or
#         script_performance["num_epochs"] != test_dict[script_idx]["num_epochs"]

# Analysis for cosntrained
higher_count = 0
lower_count = 0
higher_mag = 0
lower_mag = 0



for script_idx in dict_to_evaluate.keys():
    unconstrained_result = dict_to_evaluate[script_idx]["unconstrained"]
    target_result = dict_to_evaluate[script_idx]["constrained"]

    if unconstrained_result < target_result:
        higher_count += 1
        higher_mag += target_result - unconstrained_result

    elif unconstrained_result > target_result:
        lower_count += 1
        lower_mag += unconstrained_result - target_result
    
print(f"For simple constrained result, {higher_count} out of 210 scripts experience an increase in performance, average improvement in accuracy is {higher_mag / higher_count:.2f}%")
print(f"For simple constrained result, {lower_count} out of 210 scripts experience an decrease in performance, average decrease in accuracy is {lower_mag/ lower_count:.2f}%")


# Analysis for cosntrained
higher_count = 0
lower_count = 0
higher_mag = 0
lower_mag = 0
for script_idx in dict_to_evaluate.keys():
    unconstrained_result = dict_to_evaluate[script_idx]["unconstrained"]
    target_result = dict_to_evaluate[script_idx]["improved_constrained"]

    if unconstrained_result < target_result:
        higher_count += 1
        higher_mag += target_result - unconstrained_result

    elif unconstrained_result > target_result:
        lower_count += 1
        lower_mag += unconstrained_result - target_result
    
print(f"For simple improved_constrained result, {higher_count} out of 210 scripts experience an increase in performance, average improvement in accuracy is {higher_mag / higher_count:.2f}%")
print(f"For simple improved_constrained result, {lower_count} out of 210 scripts experience an decrease in performance, average decrease in accuracy is {lower_mag / lower_count:.2f}%")


# Analysis for cosntrained
higher_count = 0
lower_count = 0
higher_mag = 0
lower_mag = 0
for script_idx in dict_to_evaluate.keys():
    unconstrained_result = dict_to_evaluate[script_idx]["unconstrained"]
    target_result = dict_to_evaluate[script_idx]["inferred_constrained"]

    if unconstrained_result < target_result:
        higher_count += 1
        higher_mag += target_result - unconstrained_result

    elif unconstrained_result > target_result:
        lower_count += 1
        lower_mag += unconstrained_result - target_result
    
print(f"For simple inferred_constrained result, {higher_count} out of 210 scripts experience an increase in performance, average improvement in accuracy is {higher_mag / higher_count:.2f}%")
print(f"For simple inferred_constrained result, {lower_count} out of 210 scripts experience an decrease in performance, average decrease in accuracy is {lower_mag / lower_count:.2f}%")


# Analysis for cosntrained
higher_count = 0
lower_count = 0
higher_mag = 0
lower_mag = 0
for script_idx in dict_to_evaluate.keys():
    unconstrained_result = dict_to_evaluate[script_idx]["unconstrained"]
    target_result = dict_to_evaluate[script_idx]["ktrc_inferred_constrained"]

    if unconstrained_result < target_result:
        higher_count += 1
        higher_mag += target_result - unconstrained_result

    elif unconstrained_result > target_result:
        lower_count += 1
        lower_mag += unconstrained_result - target_result
    
print(f"For ktrc inferred_constrained result, {higher_count} out of 210 scripts experience an increase in performance, average improvement in accuracy is {higher_mag / higher_count:.2f}%")
print(f"For ktrc inferred_constrained result, {lower_count} out of 210 scripts experience an decrease in performance, average decrease in accuracy is {lower_mag / lower_count:.2f}%")