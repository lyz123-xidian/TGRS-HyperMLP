# nums = [2, 3, 4]
# target = 6
class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range (i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return i,j
        return []


nums = [2, 3, 4]
target = 6
Solution.twoSum(nums,target)
print(1)