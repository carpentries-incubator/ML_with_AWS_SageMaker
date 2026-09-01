---
title: "Resource Management and Monitoring"

teaching: 30
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 


- How can I monitor and manage AWS resources to avoid unnecessary costs?
- What steps are necessary to clean up SageMaker and S3 resources after the workshop?
- What best practices can help with efficient resource utilization?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand how to shut down SageMaker notebook instances to minimize costs.
- Learn to clean up S3 storage and terminate unused training jobs.
- Explore basic resource management strategies and tools for AWS.

::::::::::::::::::::::::::::::::::::::::::::::::




## Shutting down notebook instances

Notebook instances in SageMaker are billed per hour, so it’s essential to stop or delete them when they are no longer needed. Earlier in the **Notebooks as controllers** episode, we discussed using lower-cost instance types like `ml.t3.medium` (approximately $0.05/hour) for controlling workflows. While this makes open notebooks less costly than larger instances, it’s still a good habit to stop or delete notebooks to avoid unnecessary spending, especially if left idle for long periods.

1. **Navigate to SageMaker in the AWS Console.**
2. In the left-hand menu, click **Notebooks**.
3. Locate your notebook instance and select it.
4. Choose **Stop** to shut it down temporarily or **Delete** to permanently remove it.
   > **Tip:** If you plan to reuse the notebook later, stopping it is sufficient. Deleting is recommended if you are finished with the workshop.

## Cleaning up S3 storage

While S3 storage is relatively inexpensive, cleaning up unused buckets and files helps keep costs minimal and your workspace organized.

1. **Navigate to the S3 Console.**
2. Locate the bucket(s) you created for this workshop.
3. Open the bucket and select any objects (files) you no longer need.
4. Click **Delete** to remove the selected objects.
5. To delete an entire bucket:
   - Empty the bucket by selecting **Empty bucket** under **Bucket actions**.
   - Delete the bucket by clicking **Delete bucket**.

> **Reminder:** Earlier in the workshop, we set up tags for S3 buckets. Use these tags to filter and identify workshop-related buckets, ensuring that only unnecessary resources are deleted.

## Monitoring and stopping active jobs

SageMaker charges for training and tuning jobs while they run, so make sure to terminate unused jobs.

1. In the SageMaker Console, go to **Training Jobs** or **Tuning Jobs**.
2. Identify any active jobs that you no longer need.
3. Select the jobs and click **Stop**.
   > **Tip:** Review the job logs to ensure you’ve saved the results before stopping a job.

## Billing and cost monitoring

Managing your AWS expenses is vital to staying within budget. Follow these steps to monitor and control costs:

1. **Set up billing alerts:**
   - Go to the AWS **Billing Dashboard**.
   - Navigate to **Budgets** and create a budget alert to track your spending.
2. **Review usage and costs:**
   - Use the AWS **Cost Explorer** in the Billing Dashboard to view detailed expenses by service, such as SageMaker and S3.
3. **Use tags for cost tracking:**
   - Refer to the tags you set up earlier in the workshop for your notebooks and S3 buckets. These tags help you identify and monitor costs associated with specific resources.

## Best practices for resource management

Efficient resource management can save significant costs and improve your workflows. Below are some best practices:

- **Automate resource cleanup:**
   Use AWS SDK or CLI scripts to automatically shut down instances and clean up S3 buckets when not in use. [Learn more about automation with AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html).
- **Schedule resource usage:**
   Schedule instance start and stop times using AWS Lambda. [Learn how to schedule tasks with AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/tutorial-scheduled-events.html).
- **Test workflows locally first:**
   Before scaling up experiments in SageMaker, test them locally to minimize cloud usage and costs. [Learn about SageMaker local mode](https://docs.aws.amazon.com/sagemaker/latest/dg/hosting-alternatives.html).
- **Use cost tracking tools:**
   Explore AWS’s cost allocation tags and budget tracking features. [Learn more about cost management in AWS](https://aws.amazon.com/aws-cost-management/).

By following these practices and leveraging the additional resources provided, you can optimize your use of AWS while keeping costs under control.

::::::::::::::::::::::::::::::::::::: keypoints


- Always stop or delete notebook instances when not in use to avoid charges.
- Regularly clean up unused S3 buckets and objects to save on storage costs.
- Monitor your expenses through the AWS Billing Dashboard and set up alerts.
- Use tags (set up earlier in the workshop) to track and monitor costs by resource.
- Following best practices for AWS resource management can significantly reduce costs and improve efficiency.

::::::::::::::::::::::::::::::::::::::::::::::::