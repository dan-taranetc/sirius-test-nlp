from db_design import UserContext, Session


def get_context(user_id: int):
    with Session() as session:
        user_data = session.query(UserContext).filter(UserContext.user_id == user_id).first()
        if user_data:
            return user_data.context
        else:
            return {}


def update_context(user_id: int, new_message: str, new_answer: str):
    with Session() as session:
        user_data = session.query(UserContext).filter(UserContext.user_id == user_id).first()
        if user_data:
            if user_data.context:
                user_data.context['message_1'] = user_data.context['message_2']
                user_data.context['message_2'] = new_message
                user_data.context['message_3'] = new_answer
            else:
                user_data.context['message_2'] = new_message
                user_data.context['message_3'] = new_answer
            session.add(user_data)
        else:
            new_user = UserContext(
                user_id=user_id,
                context={'message_2': new_message,
                         'message_3': new_answer}
            )
            session.add(new_user)

        session.commit()


def clear_context(user_id: int):
    with Session() as session:
        user_data = session.query(UserContext).filter(UserContext.user_id == user_id).first()
        user_data.context = {}
        session.add(user_data)
        session.commit()
