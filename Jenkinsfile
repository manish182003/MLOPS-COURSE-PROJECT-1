pipeline{
    agent any

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                scripts{
                    echo 'Cloning Github repo to Jenkins................'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/manish182003/MLOPS-COURSE-PROJECT-1.git']])
                    
                }
            }
        }
    }
}