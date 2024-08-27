class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def create_football_position_tree():
    # Same tree creation code as before
    root = TreeNode("Players")
    root.left = TreeNode("Goalkeepers")
    root.right = TreeNode("Outfield Players")
    root.left.left = TreeNode("GK")
    root.right.left = TreeNode("Forwards")
    root.right.right = TreeNode("Midfielders and Defenders")
    root.right.left.left = TreeNode("ST")
    root.right.left.right = TreeNode("Wingers and CF")
    root.right.left.right.left = TreeNode("RW")
    root.right.left.right.right = TreeNode("LW")
    root.right.left.right.right.left = TreeNode("CF")
    root.right.right.left = TreeNode("Midfielders")
    root.right.right.right = TreeNode("Defenders")
    root.right.right.left.left = TreeNode("Central Midfielders")
    root.right.right.left.right = TreeNode("Wide Midfielders")
    root.right.right.left.left.left = TreeNode("CDM")
    root.right.right.left.left.right = TreeNode("CM")
    root.right.right.left.left.right.left = TreeNode("CAM")
    root.right.right.left.right.left = TreeNode("RM")
    root.right.right.left.right.right = TreeNode("LM")
    root.right.right.right.left = TreeNode("Central Defenders")
    root.right.right.right.right = TreeNode("Wide Defenders")
    root.right.right.right.left.left = TreeNode("CB")
    root.right.right.right.right.left = TreeNode("LB")
    root.right.right.right.right.right = TreeNode("RB")
    root.right.right.right.right.right.left = TreeNode("RWB")
    root.right.right.right.right.right.right = TreeNode("LWB")
    return root

def find_lca(root, pos1, pos2):
    if root is None:
        return None

    if root.value == pos1 or root.value == pos2:
        return root

    left_lca = find_lca(root.left, pos1, pos2)
    right_lca = find_lca(root.right, pos1, pos2)

    if left_lca and right_lca:
        return root

    return left_lca if left_lca else right_lca

def find_distance(root, target, distance=0):
    if root is None:
        return -1

    if root.value == target:
        return distance

    left_dist = find_distance(root.left, target, distance + 1)
    if left_dist != -1:
        return left_dist

    return find_distance(root.right, target, distance + 1)

def distance_between_positions(root, pos1, pos2):
    lca_node = find_lca(root, pos1, pos2)
    if lca_node:
        dist1 = find_distance(lca_node, pos1)
        dist2 = find_distance(lca_node, pos2)
        return dist1 + dist2
    return -1


if __name__ == '__main__':
    # Create the tree
    root = create_football_position_tree()

    # Example usage
    position1 = "RW"
    position2 = "CAM"
    distance = distance_between_positions(root, position1, position2)
    print(f"The distance between {position1} and {position2} is: {distance}")
