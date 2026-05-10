```shell
# Вывести список гитов
(online_project) PS D:\Projects\Hacker_2.0\game> git log --oneline -10                               
cbe4ff2 (HEAD -> master) Добавил принудительное закрытие соединения, при разрыве
2e0264b Базовый набросок игры
# Принудительно откатить проект к сохраненному git
(online_project) PS D:\Projects\Hacker_2.0\game> git reset --hard cbe4ff2                            
HEAD is now at cbe4ff2 Добавил принудительное закрытие соединения, при разрыве
# Отобразить файлы, которые лежат в каталоге проекта, но не соответствуют git (не отслеживались)
(online_project) PS D:\Projects\Hacker_2.0\game> git clean -fdx --dry-run
Would remove agents/
Would remove gym/
Would remove readme.md
Would remove train.py
Would remove world/
# Удалить файлы и папки
(online_project) PS D:\Projects\Hacker_2.0\game> git clean -fdx
Removing agents/
Removing gym/
Removing readme.md
Removing train.py
Removing world/
```