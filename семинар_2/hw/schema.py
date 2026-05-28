from typing import Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

# Список городов (минимум 10)
CITIES = {
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Нижний Новгород", "Самара", "Краснодар",
    "Челябинск", "Уфа", "Ростов-на-Дону", "Омск", "Казань",
    "Владивосток", "Хабаровск", "Красноярск", "Иркутск",
    "Тюмень", "Саратов", "Волгоград", "Тольятти",
    "Барнаул", "Ижевск", "Ульяновск", "Ярославль", "Томск", "Воронеж", "Калининград", "Тверь"
}

# Специальности (минимум 8)
SPECIALITIES = {
    "водитель", "инженер", "менеджер", "учитель", "врач",
    "бухгалтер", "юрист", "программист", "маркетолог",
    "дизайнер", "переводчик", "SMM", 'летчик', 'астроном', 'артист', 'музыкант'
}

# Курсы (минимум 6)
COURSES = {
    "Data Science", "Project Management", "Digital Marketing",
    "Python для анализа данных", "Управление персоналом",
    "Финансовый менеджмент", "Английский для бизнеса"
}


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=100)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[tuple(SPECIALITIES)]  # type: ignore
    desired_course: Literal[tuple(COURSES)]   # type: ignore
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    # Валидатор: возраст и год окончания не противоречат
    @field_validator("graduation_year")
    @classmethod
    def check_graduation_age(cls, v: int, info) -> int:
        age = info.data.get("age")
        if age is not None:
            current_year = datetime.now().year
            # Человек не мог окончить вуз позже, чем (текущий год - возраст + 22)
            max_graduation = current_year + age - 22
            if v > max_graduation:
                raise ValueError(f"Год окончания {v} слишком поздний для возраста {age}")
        return v

    @field_validator("address")
    @classmethod
    def city_exists(cls, v: Address) -> Address:
        if v.city not in CITIES:
            raise ValueError(f"Город «{v.city}» не в списке разрешенных")
        return v