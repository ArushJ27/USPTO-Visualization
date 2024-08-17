import json

# read original json from json file output.json
with open('output.json') as f:
    original_json = json.load(f)

# Function to convert the JSON structure iteratively
def convert_to_visualizer_format_iterative(original_json):
    root = {"name": "tsne", "children": []}  # Change the root name to "tsne"
    stack = [{"parent": root, "node": original_json}]

    while stack:
        current = stack.pop()
        parent = current["parent"]
        node = current["node"]

        if isinstance(node, dict):
            for key, value in node.items():
                new_child = {"name": key, "children": []}
                parent["children"].append(new_child)
                stack.append({"parent": new_child, "node": value})
        elif isinstance(node, list):
            for item in node:
                parent["children"].append({"name": item, "size": 1000})  # Set a default size

    return root

# Convert the original JSON
converted_json = convert_to_visualizer_format_iterative(original_json)

# Pretty print the converted JSON
#print(json.dumps(converted_json, indent=2))

# save the converted json to a new file
new_file = "converted_output.json"
with open(new_file, 'w') as f:
    json.dump(converted_json, f, indent=2)
