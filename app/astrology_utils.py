"""
Astrology calculations utility module
Handles zodiac sign calculation and basic astrological data
"""
from datetime import datetime
from typing import Dict, Optional
import pytz


class AstrologyCalculator:
    """Calculate astrological information from birth details"""
    
    ZODIAC_SIGNS = [
        ("Aries", (3, 21), (4, 19)),
        ("Taurus", (4, 20), (5, 20)),
        ("Gemini", (5, 21), (6, 20)),
        ("Cancer", (6, 21), (7, 22)),
        ("Leo", (7, 23), (8, 22)),
        ("Virgo", (8, 23), (9, 22)),
        ("Libra", (9, 23), (10, 22)),
        ("Scorpio", (10, 23), (11, 21)),
        ("Sagittarius", (11, 22), (12, 21)),
        ("Capricorn", (12, 22), (1, 19)),
        ("Aquarius", (1, 20), (2, 18)),
        ("Pisces", (2, 19), (3, 20))
    ]
    
    NAKSHATRAS = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira",
        "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha",
        "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
        "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
        "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
        "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
    ]
    
    @staticmethod
    def calculate_sun_sign(birth_date: datetime) -> str:
        """
        Calculate sun sign from birth date
        
        Args:
            birth_date: DateTime object of birth
            
        Returns:
            Sun sign name
        """
        month = birth_date.month
        day = birth_date.day
        
        for sign, (start_month, start_day), (end_month, end_day) in AstrologyCalculator.ZODIAC_SIGNS:
            # Handle signs that span year boundary (Capricorn)
            if start_month > end_month:
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day) or \
                   (month > start_month or month < end_month):
                    return sign
            else:
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day) or \
                   (start_month < month < end_month):
                    return sign
        
        return "Unknown"
    
    @staticmethod
    def calculate_moon_sign_stub(birth_date: datetime) -> str:
        """
        Stub for moon sign calculation
        In production, this would use ephemeris data
        
        Args:
            birth_date: DateTime object of birth
            
        Returns:
            Moon sign (stub - using sun sign offset)
        """
        sun_sign = AstrologyCalculator.calculate_sun_sign(birth_date)
        signs = [sign[0] for sign in AstrologyCalculator.ZODIAC_SIGNS]
        
        if sun_sign in signs:
            # Simple stub: offset by 2 signs
            idx = signs.index(sun_sign)
            moon_idx = (idx + 2) % len(signs)
            return signs[moon_idx]
        
        return "Unknown"
    
    @staticmethod
    def calculate_ascendant_stub(birth_date: datetime, birth_time: str, birth_place: str) -> str:
        """
        Stub for ascendant/rising sign calculation
        In production, this would use birth time and location coordinates
        
        Args:
            birth_date: DateTime object of birth
            birth_time: Time of birth in HH:MM format
            birth_place: Place of birth
            
        Returns:
            Ascendant sign (stub)
        """
        sun_sign = AstrologyCalculator.calculate_sun_sign(birth_date)
        signs = [sign[0] for sign in AstrologyCalculator.ZODIAC_SIGNS]
        
        if sun_sign in signs:
            # Simple stub: offset by 4 signs based on time
            try:
                hour = int(birth_time.split(":")[0])
                offset = (hour // 2) % 12
                idx = signs.index(sun_sign)
                asc_idx = (idx + offset) % len(signs)
                return signs[asc_idx]
            except:
                idx = signs.index(sun_sign)
                return signs[(idx + 4) % len(signs)]
        
        return "Unknown"
    
    @staticmethod
    def calculate_nakshatra_stub(birth_date: datetime) -> str:
        """
        Stub for nakshatra calculation
        In production, this would use moon's precise position
        
        Args:
            birth_date: DateTime object of birth
            
        Returns:
            Nakshatra name (stub)
        """
        # Simple stub based on day of year
        day_of_year = birth_date.timetuple().tm_yday
        nakshatra_idx = (day_of_year * 27 // 366) % 27
        return AstrologyCalculator.NAKSHATRAS[nakshatra_idx]
    
    @staticmethod
    def get_age(birth_date: datetime) -> int:
        """Calculate age from birth date"""
        today = datetime.now()
        age = today.year - birth_date.year
        
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return age
    
    @staticmethod
    def build_user_profile(
        name: str,
        birth_date: str,
        birth_time: str,
        birth_place: str,
        gender: Optional[str] = None,
        goals: Optional[str] = None
    ) -> Dict:
        """
        Build complete user astrological profile
        
        Args:
            name: User's name
            birth_date: Birth date in YYYY-MM-DD format
            birth_time: Birth time in HH:MM format
            birth_place: Birth place
            gender: Optional gender
            goals: Optional user goals
            
        Returns:
            Dictionary containing astrological profile
        """
        # Parse birth date
        try:
            birth_dt = datetime.strptime(birth_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid birth date format. Use YYYY-MM-DD")
        
        # Calculate astrological data
        sun_sign = AstrologyCalculator.calculate_sun_sign(birth_dt)
        moon_sign = AstrologyCalculator.calculate_moon_sign_stub(birth_dt)
        ascendant = AstrologyCalculator.calculate_ascendant_stub(birth_dt, birth_time, birth_place)
        nakshatra = AstrologyCalculator.calculate_nakshatra_stub(birth_dt)
        age = AstrologyCalculator.get_age(birth_dt)
        
        profile = {
            "name": name,
            "birth_date": birth_date,
            "birth_time": birth_time,
            "birth_place": birth_place,
            "age": age,
            "sun_sign": sun_sign,
            "moon_sign": moon_sign,
            "ascendant": ascendant,
            "nakshatra": nakshatra
        }
        
        if gender:
            profile["gender"] = gender
        
        if goals:
            profile["goals"] = goals
        
        return profile


# Example usage
if __name__ == "__main__":
    profile = AstrologyCalculator.build_user_profile(
        name="Ritika",
        birth_date="1995-08-20",
        birth_time="14:30",
        birth_place="Jaipur, India"
    )
    print("User Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
